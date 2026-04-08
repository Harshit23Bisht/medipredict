"""
api/main.py
-----------
Main FastAPI application.
Loads XGBoost + LSTM models at startup.
"""

import os
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── Model Store ───────────────────────────────────────────────
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load XGBoost
    xgb_path = "data/models/xgboost_mimic_v1.pkl"
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            models["xgboost"] = pickle.load(f)
        print(f"[startup] XGBoost loaded from {xgb_path}")
    else:
        print(f"[startup] WARNING: XGBoost model not found at {xgb_path}")

    # Load LSTM
    lstm_path = "data/models/lstm_mimic_v1.pkl"
    if os.path.exists(lstm_path):
        with open(lstm_path, "rb") as f:
            models["lstm"] = pickle.load(f)
        print(f"[startup] LSTM loaded from {lstm_path}")
    else:
        print(f"[startup] LSTM model not found yet — XGBoost only mode")

    yield
    models.clear()

# ── App ───────────────────────────────────────────────────────
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