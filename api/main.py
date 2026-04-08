"""
api/main.py
-----------
Main FastAPI application.
Loads XGBoost + LSTM + CNN (optional) models at startup.

Fix applied vs original:
  - torchvision.models renamed to tv_models to avoid collision
    with the global `models` dict (was silently overwriting it)
  - CNN loading wrapped in try/except so API starts even without .pth
  - CNN stored in models["cnn"] dict for access by routers
"""

import os
import pickle
from contextlib import asynccontextmanager

import torch
import torchvision.models as tv_models          # renamed — avoids dict collision

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── Global model store ────────────────────────────────────────
# All routers import this dict via `import api.main as app_state`
# and call app_state.models.get("xgboost") etc.
models = {}


# ─────────────────────────────────────────────────────────────
def _load_xgboost():
    path = "data/models/xgboost_mimic_v1.pkl"
    if not os.path.exists(path):
        print(f"[startup] WARNING: XGBoost model not found at {path}")
        return
    with open(path, "rb") as f:
        models["xgboost"] = pickle.load(f)
    print(f"[startup] XGBoost loaded  ← {path}  "
          f"(AUC={models['xgboost'].get('test_auc', '?'):.4f})")


def _load_lstm():
    path = "data/models/lstm_mimic_v1.pkl"
    if not os.path.exists(path):
        print(f"[startup] LSTM not found — XGBoost-only mode")
        return
    with open(path, "rb") as f:
        models["lstm"] = pickle.load(f)
    print(f"[startup] LSTM loaded     ← {path}  "
          f"(AUC={models['lstm'].get('test_auc', '?'):.4f})")


def _load_cnn():
    """
    Loads teammate's ResNet50 CNN from cnn_model.pth.

    Teammate specs:
      - Input : 224×224 RGB, ImageNet normalisation
      - Output : single sigmoid probability (already in [0,1])
      - Architecture: ResNet50 with fc replaced by Linear(2048, 1)
                      followed by Sigmoid in the model itself.

    We do NOT apply sigmoid again in fusion — teammate's model does it.
    """
    path = "data/models/cnn_model.pth"
    if not os.path.exists(path):
        print(f"[startup] CNN not found at {path} — will use XGB+LSTM only")
        return

    try:
        # Reconstruct the same architecture teammate used
        cnn_net = tv_models.resnet50(weights=None)       # no pretrained weights
        cnn_net.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid(),                          # teammate outputs probability
        )
        cnn_net.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        cnn_net.eval()

        models["cnn"] = {
            "model":   cnn_net,
            "version": "resnet50_nih_v1",
            # Normalisation constants — used by predict router when
            # preprocessing the uploaded X-ray before inference
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
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

# ── Routers ───────────────────────────────────────────────────
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