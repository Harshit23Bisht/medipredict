"""
models/xgboost_model.py
-----------------------
Trains XGBoost on real MIMIC-IV data.
Features : encounter_features view (11 columns)
Label    : readmission_label view  (readmitted_30d)
Split    : chronological 70 / 15 / 15
Run      : python models/xgboost_model.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve,
)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from dotenv import load_dotenv
import shap

load_dotenv()

DB_URL = os.getenv(
    "POSTGRES_URL",
    os.getenv("DATABASE_URL",
              "postgresql://postgres:hb23@localhost:5432/medipredict"),
)
engine = create_engine(DB_URL, pool_pre_ping=True)

os.makedirs("data/models", exist_ok=True)
os.makedirs("eda_plots",   exist_ok=True)

FEATURE_COLS = [
    "age_at_admission",
    "gender",
    "length_of_stay",
    "num_diagnoses",
    "num_medications",
    "max_creatinine",
    "max_wbc",
    "num_prior_admissions",
]

# ─────────────────────────────────────────────────────────────
# 0. Data Quality Diagnostic
# ─────────────────────────────────────────────────────────────
def run_diagnostic():
    """
    Prints row counts at each stage so you can see exactly where
    encounters are being lost.  Run once — if encounter_features
    << encounter table, the view is filtering too aggressively
    (likely INNER JOINs to vital_sign / lab_result that drop
    encounters with no ICU data).  Switch those to LEFT JOINs
    in the view definition.
    """
    checks = {
        "encounter table":        "SELECT COUNT(*) FROM encounter",
        "encounter_features view":"SELECT COUNT(*) FROM encounter_features",
        "readmission_label view": "SELECT COUNT(*) FROM readmission_label",
        "readmitted_30d = 1":     "SELECT COUNT(*) FROM readmission_label WHERE readmitted_30d = 1",
        "after JOIN + LOS > 0":  """
            SELECT COUNT(*) FROM encounter_features ef
            JOIN readmission_label rl ON rl.encounter_id = ef.encounter_id
            WHERE ef.length_of_stay > 0
        """,
    }
    print("\n── Data Quality Diagnostic ──────────────────────────")
    with engine.connect() as conn:
        for label, q in checks.items():
            n = conn.execute(text(q)).scalar()
            print(f"  {label:<30} : {n:>10,}")
    print()
    print("  ⚠️  If 'encounter_features view' << 'encounter table',")
    print("       open the view in pgAdmin and change INNER JOINs")
    print("       to LEFT JOINs on vital_sign and lab_result.")
    print("────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Loading data from PostgreSQL...")
    query = text("""
        SELECT
            ef.encounter_id,
            ef.admit_date,
            ef.age_at_admission,
            ef.gender,
            ef.length_of_stay,
            ef.num_diagnoses,
            ef.num_medications,
            ef.avg_hr,
            ef.max_bp_sys,
            ef.avg_temp,
            ef.max_creatinine,
            ef.max_wbc,
            ef.num_prior_admissions,
            rl.readmitted_30d AS label
        FROM encounter_features ef
        JOIN readmission_label rl
          ON rl.encounter_id = ef.encounter_id
        WHERE ef.length_of_stay > 0
        ORDER BY ef.admit_date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        # 🔥 FIX: better missing handling
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    print(f"  Total encounters : {len(df):,}")
    print(f"  Readmission rate : {df['label'].mean():.2%}")
    print(f"  Date range       : {df['admit_date'].min()} → {df['admit_date'].max()}")

    # ── Warn if data looks suspicious ────────────────────────
    if len(df) < 50_000:
        print("\n  ⚠️  WARNING: Only {:,} rows loaded (expected ~245k).".format(len(df)))
        print("     Run run_diagnostic() to find where rows are lost.")
        print("     Most likely fix: change INNER JOINs → LEFT JOINs in")
        print("     the encounter_features view definition.\n")

    if df['label'].mean() < 0.05:
        print(f"\n  ⚠️  WARNING: Readmission rate is {df['label'].mean():.2%} (expected ~20%).")
        print("     The readmission_label view may not be computing correctly.")
        print("     Verify the LEAD window function spans all encounters,")
        print("     not just the filtered subset.\n")

    return df


# ─────────────────────────────────────────────────────────────
# 2. Chronological split
# ─────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    n         = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:5s}: {len(part):,}  "
              f"({part['admit_date'].min()} → {part['admit_date'].max()})  "
              f"readmit={part['label'].mean():.2%}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────
def train_model(train: pd.DataFrame, val: pd.DataFrame):
    print("\nTraining XGBoost...")

    # ✅ Use ORIGINAL data (NO UPSAMPLING)
    X_train = train[FEATURE_COLS].fillna(0).astype(float)
    y_train = train["label"].astype(int)

    X_val = val[FEATURE_COLS].fillna(0).astype(float)
    y_val = val["label"].astype(int)

    # ✅ Proper class imbalance handling
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_wt = neg / pos if pos > 0 else 1.0

    print(f"  scale_pos_weight : {scale_pos_wt:.2f}  "
          f"(pos={pos:,}  neg={neg:,})")

    # ✅ Stable model (prevents overfitting on weak data)
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_wt,
        eval_metric="auc",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"  Best iteration   : {model.best_iteration}")
    return model, X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, X_train, y_train):
    print("\n" + "=" * 52)
    print("MODEL EVALUATION")
    print("=" * 52)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob  = model.predict_proba(X_test)[:, 1]
    test_pred  = model.predict(X_test)

    train_auc  = roc_auc_score(y_train, train_prob)
    test_auc   = roc_auc_score(y_test,  test_prob)
    test_prauc = average_precision_score(y_test, test_prob)
    gap        = train_auc - test_auc

    print(f"  Train ROC-AUC : {train_auc:.4f}")
    print(f"  Test  ROC-AUC : {test_auc:.4f}")
    print(f"  Test  PR-AUC  : {test_prauc:.4f}")
    print(f"  Overfit gap   : {gap:.4f}  "
          + ("✅" if gap < 0.05 else "⚠️  possible overfit"))

    print("\nClassification Report:")
    print(classification_report(
        y_test, test_pred,
        target_names=["Not Readmitted", "Readmitted"],
    ))

    print("Feature importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    return test_auc, test_prauc, test_prob, test_pred


# ─────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────
def save_plots(model, X_train, y_train,
               X_test, y_test,
               test_prob, test_pred, test_auc):
    print("\nSaving plots → eda_plots/ ...")

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2196F3", lw=2.5,
             label=f"XGBoost (AUC={test_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — XGBoost on MIMIC-IV", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_plots/roc_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Not Readmitted", "Readmitted"]
    ).plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — XGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_plots/confusion_matrix.png", dpi=150)
    plt.close()

    feat_imp = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    feat_imp.plot(kind="barh", color="#2196F3", edgecolor="white")
    plt.title("Feature Importances — XGBoost", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("eda_plots/feature_importance.png", dpi=150)
    plt.close()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    try:
        dummy_auc = roc_auc_score(
            y_test, dummy.predict_proba(X_test)[:, 1])
    except Exception:
        dummy_auc = 0.5

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Baseline\n(No Features)", "XGBoost\n(With Features)"],
        [dummy_auc, test_auc],
        color=["#9E9E9E", "#4CAF50"],
        edgecolor="white", width=0.4,
    )
    plt.ylim(0, 1.15)
    for bar, val in zip(bars, [dummy_auc, test_auc]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02, f"{val:.4f}",
            ha="center", fontweight="bold", fontsize=13,
        )
    plt.title("Baseline vs XGBoost — ROC-AUC", fontweight="bold")
    plt.ylabel("ROC-AUC Score")
    plt.tight_layout()
    plt.savefig("eda_plots/model_comparison.png", dpi=150)
    plt.close()

    print("  Generating SHAP values...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        plt.figure()
        shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance", fontweight="bold")
        plt.tight_layout()
        plt.savefig("eda_plots/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP plot saved ✅")
    except Exception as exc:
        print(f"  SHAP skipped: {exc}")

    print("  All plots saved ✅")


# ─────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────
def save_model(model, test_auc: float, test_prauc: float):
    path = "data/models/xgboost_mimic_v1.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "model":        model,
                "feature_cols": FEATURE_COLS,
                "version":      "xgb_mimic_v1",
                "test_auc":     test_auc,
                "test_prauc":   test_prauc,
            },
            fh,
        )
    print(f"\nModel saved → {path}")


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("MediPredict — XGBoost Training")
    print("=" * 52)

    # Always run diagnostic first so problems are visible
    run_diagnostic()

    df = load_data()

    print("\nSplitting (chronological 70/15/15)...")
    train, val, test = split_data(df)

    X_test = test[FEATURE_COLS].fillna(0).astype(float)
    y_test = test["label"].astype(int)

    model, X_train, y_train, X_val, y_val = train_model(train, val)

    test_auc, test_prauc, test_prob, test_pred = evaluate(
        model, X_test, y_test, X_train, y_train
    )

    save_plots(
        model, X_train, y_train,
        X_test, y_test,
        test_prob, test_pred, test_auc,
    )

    save_model(model, test_auc, test_prauc)

    print("\n✅ XGBoost training complete!")
    print(f"   ROC-AUC : {test_auc:.4f}")
    print(f"   PR-AUC  : {test_prauc:.4f}")