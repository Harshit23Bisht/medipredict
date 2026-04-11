"""
api/main.py
-----------
Main FastAPI application.
Loads XGBoost + LSTM + CNN (optional) models at startup.

Patches vs your original:
  - _load_xgboost: wraps model in {"model": ..., "test_auc": ...} dict
    so predict.py can do app_state.models["xgboost"]["model"].predict_proba()
  - _load_lstm: same dict wrapping, also handles torch-saved .pkl files
  - _load_cnn: corrected filename to cnn_v1.pth, added state_dict fallback
    for teammates who torch.save(model.state_dict()) vs torch.save(model)
  - Everything else is identical to your original
"""

import os
import pickle
from contextlib import asynccontextmanager
import sys
import torch
import torchvision.models as tv_models

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from models.lstm_model import BiLSTM
sys.modules["__main__"].BiLSTM = BiLSTM
load_dotenv()

# ── Global model store ────────────────────────────────────────
models = {}


def _load_xgboost():
    path = "data/models/xgboost_mimic_v1.pkl"
    if not os.path.exists(path):
        print(f"[startup] WARNING: XGBoost model not found at {path}")
        return
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # pkl may be the raw model OR a dict {"model": ..., "test_auc": ...}
    if isinstance(obj, dict) and "model" in obj:
        models["xgboost"] = obj
    else:
        models["xgboost"] = {"model": obj, "test_auc": None}

    auc = models["xgboost"].get("test_auc")
    auc_str = f"{auc:.4f}" if auc is not None else "?"
    print(f"[startup] XGBoost loaded  ← {path}  (AUC={auc_str})")


def _load_lstm():
    path = "data/models/lstm_mimic_v1.pkl"
    if not os.path.exists(path):
        print("[startup] LSTM not found — XGBoost-only mode")
        return

    try:
        # Try torch.load first (handles both .pkl and .pth saved via torch)
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = None

    if obj is None:
        # Fall back to pickle
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[startup] LSTM load FAILED: {e}")
            return

    # Normalise to dict format expected by predict.py
    if isinstance(obj, dict) and "model" in obj:
        lstm_entry = obj
        m = lstm_entry["model"]
        if isinstance(m, torch.nn.Module):
            m.eval()
    elif isinstance(obj, torch.nn.Module):
        obj.eval()
        lstm_entry = {
            "model":      obj,
            "test_auc":   None,
            "seq_length": 48,
            "vital_cols": ["heart_rate", "bp_systolic", "bp_diastolic",
                           "temperature_f", "respiratory_rate", "spo2"],
            "scaler":     None,
        }
    else:
        # Raw sklearn-like object (unlikely but possible)
        lstm_entry = {"model": obj, "test_auc": None, "seq_length": 48,
                      "vital_cols": ["heart_rate","bp_systolic","bp_diastolic",
                                     "temperature_f","respiratory_rate","spo2"],
                      "scaler": None}

    models["lstm"] = lstm_entry
    auc = lstm_entry.get("test_auc")
    auc_str = f"{auc:.4f}" if auc is not None else "?"
    print(f"[startup] LSTM loaded     ← {path}  (AUC={auc_str})")


def _load_cnn():
    """
    Loads teammate's ResNet50 CNN.

    Teammate specs:
      - Input : 224×224 RGB, ImageNet normalisation
      - Output : single sigmoid probability (already in [0,1])
      - Architecture: ResNet50 with fc = Linear(2048,1) + Sigmoid

    We do NOT apply sigmoid again in fusion.
    """
    # Support both filenames in case teammate uses either
    for candidate in ["data/models/cnn_v1.pth", "data/models/cnn_model.pth"]:
        if os.path.exists(candidate):
            path = candidate
            break
    else:
        print("[startup] CNN not found — will use XGB+LSTM only")
        return

    try:
        cnn_net = tv_models.resnet50(weights=None)
        cnn_net.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),
        )

        state = torch.load(path, map_location="cpu")

        if isinstance(state, torch.nn.Module):
            # Teammate did torch.save(model) — use it directly
            cnn_net = state
        elif isinstance(state, dict) and "state_dict" in state:
            cnn_net.load_state_dict(state["state_dict"])
        else:
            # Raw state dict
            try:
                cnn_net.load_state_dict(state)
            except RuntimeError:
                cnn_net.load_state_dict(state, strict=False)
                print("[startup] CNN loaded with strict=False — some weights missing")

        cnn_net.eval()
        models["cnn"] = {
            "model":   cnn_net,
            "version": "resnet50_nih_v1",
            "mean":    [0.485, 0.456, 0.406],
            "std":     [0.229, 0.224, 0.225],
        }
        print(f"[startup] CNN loaded      ← {path}")

    except Exception as exc:
        print(f"[startup] CNN load FAILED: {exc}")
        print("          API will start in XGB+LSTM mode.")


# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_xgboost()
    _load_lstm()
    _load_cnn()

    loaded = list(models.keys())
    print(f"[startup] Active models: {loaded}")
    if "xgboost" not in loaded:
        print("[startup] ⚠️  No XGBoost model — predictions will fail.")

    yield

    models.clear()
    print("[shutdown] Models cleared.")


# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MediPredict API",
    description = "30-day hospital readmission risk prediction — MIMIC-IV",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

from api.routers import patients, encounters, predict, labs

app.include_router(patients.router,   prefix="/patients",   tags=["Patients"])
app.include_router(encounters.router, prefix="/encounters", tags=["Encounters"])
app.include_router(predict.router,    prefix="/predict",    tags=["Predictions"])
app.include_router(labs.router,       prefix="/labs",       tags=["Labs"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "MediPredict API v2",
        "models":  list(models.keys()),
    }