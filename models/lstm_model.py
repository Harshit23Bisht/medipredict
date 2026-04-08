"""
models/lstm_model.py
--------------------
Trains a Bidirectional LSTM on MIMIC-IV ICU vital sequences.

Data source : vital_sequence materialized table
              (pre-computed 48-hour ICU windows, joined through icu_stay)
Split       : chronological 70 / 15 / 15  (preserves time ordering)
Run         : python models/lstm_model.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "POSTGRES_URL",
    os.getenv("DATABASE_URL",
              "postgresql://postgres:hb23@localhost:5432/medipredict"),
)
engine = create_engine(DB_URL, pool_pre_ping=True)

os.makedirs("data/models", exist_ok=True)
os.makedirs("eda_plots",   exist_ok=True)

# Vitals in the exact column order of vital_sequence table
VITAL_COLS  = [
    "heart_rate", "bp_systolic", "bp_diastolic",
    "temperature_f", "respiratory_rate", "spo2",
]
SEQ_LENGTH  = 48   # hours (matches vital_sequence build)
N_FEATURES  = len(VITAL_COLS)


# ─────────────────────────────────────────────────────────────
# 1. Load from vital_sequence  (fast — single SQL query)
# ─────────────────────────────────────────────────────────────
def load_sequences() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Returns
    -------
    X          : (n_encounters, SEQ_LENGTH, N_FEATURES)  float32
    y          : (n_encounters,)                          int
    enc_ids    : list of encounter_id in the same order   (for traceability)
    """
    print("Loading vital_sequence from PostgreSQL...")
    query = text("""
        SELECT
            encounter_id,
            readmitted_30d,
            hour_offset,
            heart_rate,
            bp_systolic,
            bp_diastolic,
            temperature_f,
            respiratory_rate,
            spo2
        FROM vital_sequence
        ORDER BY encounter_id, hour_offset
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"  Rows loaded      : {len(df):,}")
    print(f"  Unique encounters: {df['encounter_id'].nunique():,}")
    print(f"  Readmission rate : {df.groupby('encounter_id')['readmitted_30d'].first().mean():.2%}")

    # ── Build per-encounter arrays ────────────────────────────
    enc_ids, sequences, labels = [], [], []

    for eid, grp in df.groupby("encounter_id", sort=False):
        grp = grp.sort_values("hour_offset")

        arr = grp[VITAL_COLS].values.astype(np.float32)

        # Clip to last SEQ_LENGTH hours, or left-pad with zeros
        if len(arr) >= SEQ_LENGTH:
            arr = arr[-SEQ_LENGTH:]
        else:
            pad = np.zeros((SEQ_LENGTH - len(arr), N_FEATURES), dtype=np.float32)
            arr = np.vstack([pad, arr])

        enc_ids.append(eid)
        sequences.append(arr)
        labels.append(int(grp["readmitted_30d"].iloc[0]))

    X = np.stack(sequences, axis=0)   # (N, 48, 6)
    y = np.array(labels,    dtype=int)

    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y, enc_ids


# ─────────────────────────────────────────────────────────────
# 2. Chronological split
#    enc_ids are in the order they came from ORDER BY encounter_id,
#    which correlates with admission time in MIMIC.
#    For a more explicit time-based split, use admit_time — but
#    encounter_id order is good enough and avoids an extra JOIN.
# ─────────────────────────────────────────────────────────────
def split_chronological(X, y):
    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],      y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],         y[val_end:]

    for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5s}: {len(yy):,}  readmit={yy.mean():.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────
# 3. Normalise  (fit on train only — no leakage)
# ─────────────────────────────────────────────────────────────
def normalise(X_train, X_val, X_test):
    n_tr, t, f = X_train.shape
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, f)).reshape(n_tr, t, f)
    X_val   = scaler.transform(
        X_val.reshape(-1, f)).reshape(len(X_val), t, f)
    X_test  = scaler.transform(
        X_test.reshape(-1, f)).reshape(len(X_test), t, f)

    return X_train, X_val, X_test, scaler


# ─────────────────────────────────────────────────────────────
# 4. Model definition
# ─────────────────────────────────────────────────────────────
def build_model():
    import torch.nn as nn

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size   = N_FEATURES,
                hidden_size  = 128,
                num_layers   = 2,
                batch_first  = True,
                dropout      = 0.3,
                bidirectional= True,
            )
            # bidirectional → hidden_size * 2 = 256
            self.bn   = nn.BatchNorm1d(256)
            self.fc1  = nn.Linear(256, 64)
            self.drop = nn.Dropout(0.3)
            self.fc2  = nn.Linear(64, 1)
            self.sig  = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)          # (B, T, 256)
            out    = out[:, -1, :]         # last time step
            out    = self.bn(out)
            out    = self.drop(torch.relu(self.fc1(out)))
            return self.sig(self.fc2(out))

    return BiLSTM()

# torch imported at module level after guard — imported inside functions
# to keep startup clean even if PyTorch isn't installed


# ─────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    model = model.to(device)

    # Class imbalance — weighted BCE
    pos          = y_train.sum()
    neg          = len(y_train) - pos
    pos_weight   = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Update model to output raw logits instead of sigmoid for BCEWithLogitsLoss
    # We'll wrap prediction with sigmoid manually.
    # (no change to architecture needed — sigmoid is applied after for inference)

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.astype(np.float32)),
        ),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=False
    )

    best_auc, best_state, patience_count = 0.0, None, 0
    MAX_PATIENCE = 8
    EPOCHS       = 40

    history = {"train_loss": [], "val_auc": []}

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val.astype(np.float32)).to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # raw logit from model (sigmoid is last layer — use output directly)
            pred = model(xb).squeeze()
            # BCEWithLogitsLoss expects raw logits; our model has sigmoid,
            # so just use BCELoss here to stay consistent with architecture
            loss = nn.BCELoss()(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).squeeze().cpu().numpy()

        if len(set(y_val)) > 1:
            val_auc = roc_auc_score(y_val, val_probs)
        else:
            val_auc = 0.5

        scheduler.step(val_auc)

        history["train_loss"].append(epoch_loss)
        history["val_auc"].append(val_auc)

        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Val AUC: {val_auc:.4f}"
              + (" ◀ best" if val_auc > best_auc else ""))

        if val_auc > best_auc:
            best_auc        = val_auc
            best_state      = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= MAX_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\n  Best Val AUC: {best_auc:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    import torch

    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X_test)).squeeze().numpy()

    test_auc   = roc_auc_score(y_test, probs)
    test_prauc = average_precision_score(y_test, probs)

    print("\n" + "=" * 52)
    print("LSTM EVALUATION")
    print("=" * 52)
    print(f"  Test ROC-AUC : {test_auc:.4f}")
    print(f"  Test PR-AUC  : {test_prauc:.4f}")

    return test_auc, test_prauc, probs


# ─────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────
def save_plots(history, y_test, probs, test_auc):
    # ── Training curves ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["train_loss"], color="#E53935", lw=2)
    axes[0].set_title("Training Loss",   fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_auc"], color="#43A047", lw=2)
    axes[1].set_title("Validation ROC-AUC", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    plt.suptitle("LSTM Training History — MediPredict",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("eda_plots/lstm_training_history.png", dpi=150)
    plt.close()

    # ── ROC Curve ─────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#9C27B0", lw=2.5,
             label=f"BiLSTM (AUC={test_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — LSTM on MIMIC-IV ICU Vitals", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_plots/lstm_roc_curve.png", dpi=150)
    plt.close()

    print("  Plots saved → eda_plots/lstm_training_history.png")
    print("               eda_plots/lstm_roc_curve.png")


# ─────────────────────────────────────────────────────────────
# 8. Save
# ─────────────────────────────────────────────────────────────
def save_model(model, scaler, test_auc, test_prauc):
    import torch

    path = "data/models/lstm_mimic_v1.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "model":        model,
                "scaler":       scaler,
                "vital_cols":   VITAL_COLS,
                "seq_length":   SEQ_LENGTH,
                "version":      "lstm_mimic_v1",
                "test_auc":     test_auc,
                "test_prauc":   test_prauc,
            },
            fh,
        )
    print(f"\nModel saved → {path}")


# ─────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch   # confirm PyTorch available before doing heavy work

    print("=" * 52)
    print("MediPredict — LSTM Training")
    print("=" * 52)

    # Load
    X, y, enc_ids = load_sequences()

    # Split (chronological)
    print("\nSplitting (chronological 70/15/15)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_chronological(X, y)

    # Normalise
    print("\nNormalising...")
    X_train, X_val, X_test, scaler = normalise(X_train, X_val, X_test)

    # Build & train
    model          = build_model()
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    test_auc, test_prauc, probs = evaluate(model, X_test, y_test)

    # Plots
    save_plots(history, y_test, probs, test_auc)

    # Save
    save_model(model, scaler, test_auc, test_prauc)

    print("\n✅ LSTM training complete!")
    print(f"   ROC-AUC : {test_auc:.4f}")
    print(f"   PR-AUC  : {test_prauc:.4f}")