"""
api/routers/predict.py
----------------------
Prediction endpoints.

Endpoints:
  POST /predict/{encounter_id}              — XGB + LSTM (+ CNN if model loaded)
  POST /predict/{encounter_id}/with-image   — same, but accepts X-ray upload
  GET  /predict/history/{encounter_id}      — past predictions for an encounter
  GET  /predict/stats/summary               — aggregate stats
"""

import io
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import get_db
import api.main as app_state

router = APIRouter()

FEATURE_COLS = [
    "age_at_admission", "gender", "length_of_stay",
    "num_diagnoses", "num_medications", "avg_hr",
    "max_bp_sys", "avg_temp", "max_creatinine",
    "max_wbc", "num_prior_admissions",
]

# ImageNet normalisation — must match teammate's training pipeline
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def risk_level(score: float) -> str:
    if score < 0.30:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    return "HIGH"


def get_xgb_score(encounter_id: int, db: Session) -> dict:
    row = db.execute(text("""
        SELECT
            age_at_admission, gender, length_of_stay,
            num_diagnoses, num_medications, avg_hr,
            max_bp_sys, avg_temp, max_creatinine,
            max_wbc, num_prior_admissions
        FROM encounter_features
        WHERE encounter_id = :eid
    """), {"eid": encounter_id}).fetchone()

    if not row:
        raise HTTPException(404, f"No features found for encounter {encounter_id}")

    xgb_data = app_state.models.get("xgboost")
    if not xgb_data:
        raise HTTPException(503, "XGBoost model not loaded")

    features = dict(row._mapping)
    X        = pd.DataFrame([features])[FEATURE_COLS].fillna(0).astype(float)
    score    = float(xgb_data["model"].predict_proba(X)[0, 1])
    return {"score": score, "features": features}


def get_lstm_score(encounter_id: int, db: Session) -> float | None:
    lstm_data = app_state.models.get("lstm")
    if not lstm_data:
        return None

    # Use vital_sequence (ICU encounters) — faster and pre-computed
    rows = db.execute(text("""
        SELECT hour_offset, heart_rate, bp_systolic, bp_diastolic,
               temperature_f, respiratory_rate, spo2
        FROM vital_sequence
        WHERE encounter_id = :eid
        ORDER BY hour_offset
    """), {"eid": encounter_id}).fetchall()

    # Fallback: aggregate raw vitals by hour
    if not rows:
        rows = db.execute(text("""
            SELECT
                AVG(CASE WHEN vital_type='heart_rate'       THEN value END) AS heart_rate,
                AVG(CASE WHEN vital_type='bp_systolic'      THEN value END) AS bp_systolic,
                AVG(CASE WHEN vital_type='bp_diastolic'     THEN value END) AS bp_diastolic,
                AVG(CASE WHEN vital_type='temperature_f'    THEN value END) AS temperature_f,
                AVG(CASE WHEN vital_type='respiratory_rate' THEN value END) AS respiratory_rate,
                AVG(CASE WHEN vital_type='spo2'             THEN value END) AS spo2
            FROM vital_sign
            WHERE encounter_id = :eid
            AND vital_type IN (
                'heart_rate','bp_systolic','bp_diastolic',
                'temperature_f','respiratory_rate','spo2'
            )
            GROUP BY DATE_TRUNC('hour', recorded_at)
            ORDER BY DATE_TRUNC('hour', recorded_at) DESC
            LIMIT :sl
        """), {"eid": encounter_id, "sl": lstm_data.get("seq_length", 48)}).fetchall()

    if not rows:
        return None

    try:
        seq_len  = lstm_data.get("seq_length", 48)
        vitals   = lstm_data.get("vital_cols",
                                 ["heart_rate","bp_systolic","bp_diastolic",
                                  "temperature_f","respiratory_rate","spo2"])
        scaler   = lstm_data.get("scaler")
        model    = lstm_data["model"]

        df  = pd.DataFrame([dict(r._mapping) for r in rows])
        arr = df[vitals].fillna(0).values.astype(np.float32)

        if len(arr) >= seq_len:
            arr = arr[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(arr), len(vitals)), dtype=np.float32)
            arr = np.vstack([pad, arr])

        if scaler:
            arr = scaler.transform(arr)

        tensor = torch.FloatTensor(arr).unsqueeze(0)   # (1, T, F)

        model.eval()
        with torch.no_grad():
            logit = model(tensor).squeeze()
            # Model outputs logits (sigmoid removed from architecture)
            # Apply sigmoid here to get probability
            prob  = torch.sigmoid(logit).item()

        return float(prob)

    except Exception as e:
        print(f"LSTM scoring error for encounter {encounter_id}: {e}")
        return None


def get_cnn_score(image_bytes: bytes | None) -> float | None:
    """
    Score a chest X-ray with the CNN.

    Teammate's model already applies sigmoid internally, so the output
    is already a probability in [0, 1]. Do NOT apply sigmoid again.
    """
    if image_bytes is None:
        return None

    cnn_data = app_state.models.get("cnn")
    if not cnn_data:
        return None

    try:
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = CNN_TRANSFORM(img).unsqueeze(0)        # (1, 3, 224, 224)

        cnn_model = cnn_data["model"]
        cnn_model.eval()
        with torch.no_grad():
            # Teammate's model has Sigmoid as last layer → output is probability
            score = cnn_model(tensor).squeeze().item()

        return float(score)

    except Exception as e:
        print(f"CNN scoring error: {e}")
        return None


def compute_fusion(xgb_score: float,
                   lstm_score: float | None,
                   cnn_score:  float | None) -> tuple[float, dict]:
    """
    Weighted fusion. Redistributes weight from missing models.
    Returns (fusion_score, weights_used).
    """
    has_lstm = lstm_score is not None
    has_cnn  = cnn_score  is not None

    if has_lstm and has_cnn:
        w = {"xgb": 0.50, "lstm": 0.30, "cnn": 0.20}
        score = w["xgb"]*xgb_score + w["lstm"]*lstm_score + w["cnn"]*cnn_score

    elif has_lstm:
        w = {"xgb": 0.60, "lstm": 0.40, "cnn": 0.00}
        score = w["xgb"]*xgb_score + w["lstm"]*lstm_score

    elif has_cnn:
        w = {"xgb": 0.70, "lstm": 0.00, "cnn": 0.30}
        score = w["xgb"]*xgb_score + w["cnn"]*cnn_score

    else:
        w = {"xgb": 1.00, "lstm": 0.00, "cnn": 0.00}
        score = xgb_score

    return round(score, 4), w


def _save_prediction(db: Session, encounter_id: int,
                     xgb_score, lstm_score, cnn_score, fusion_score):
    db.execute(text("""
        INSERT INTO risk_prediction
            (encounter_id, xgb_score, lstm_score, cnn_score,
             fusion_score, risk_level, model_version)
        VALUES
            (:eid, :xgb, :lstm, :cnn, :fusion, :level, :ver)
    """), {
        "eid":    encounter_id,
        "xgb":    round(xgb_score,   4),
        "lstm":   round(lstm_score,  4) if lstm_score is not None else None,
        "cnn":    round(cnn_score,   4) if cnn_score  is not None else None,
        "fusion": round(fusion_score,4),
        "level":  risk_level(fusion_score),
        "ver":    "v2_mimic",
    })
    db.commit()


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/{encounter_id}")
def predict(encounter_id: int, db: Session = Depends(get_db)):
    """Predict readmission risk using XGBoost + LSTM (no image)."""
    xgb_result   = get_xgb_score(encounter_id, db)
    xgb_score    = xgb_result["score"]
    lstm_score   = get_lstm_score(encounter_id, db)
    fusion_score, weights = compute_fusion(xgb_score, lstm_score, None)

    _save_prediction(db, encounter_id, xgb_score, lstm_score, None, fusion_score)

    return {
        "encounter_id":  encounter_id,
        "xgb_score":     round(xgb_score,  4),
        "lstm_score":    round(lstm_score, 4) if lstm_score is not None else None,
        "cnn_score":     None,
        "fusion_score":  fusion_score,
        "risk_level":    risk_level(fusion_score),
        "weights_used":  weights,
        "model_version": "v2_mimic",
        "features_used": xgb_result["features"],
    }


@router.post("/{encounter_id}/with-image")
async def predict_with_image(
    encounter_id: int,
    xray: UploadFile = File(..., description="Chest X-ray (JPEG/PNG)"),
    db: Session = Depends(get_db),
):
    """Predict readmission risk using XGBoost + LSTM + CNN (chest X-ray upload)."""
    image_bytes  = await xray.read()
    xgb_result   = get_xgb_score(encounter_id, db)
    xgb_score    = xgb_result["score"]
    lstm_score   = get_lstm_score(encounter_id, db)
    cnn_score    = get_cnn_score(image_bytes)
    fusion_score, weights = compute_fusion(xgb_score, lstm_score, cnn_score)

    _save_prediction(db, encounter_id, xgb_score, lstm_score, cnn_score, fusion_score)

    return {
        "encounter_id":  encounter_id,
        "xgb_score":     round(xgb_score,  4),
        "lstm_score":    round(lstm_score, 4) if lstm_score is not None else None,
        "cnn_score":     round(cnn_score,  4) if cnn_score  is not None else None,
        "fusion_score":  fusion_score,
        "risk_level":    risk_level(fusion_score),
        "weights_used":  weights,
        "model_version": "v2_mimic",
        "features_used": xgb_result["features"],
    }


@router.get("/history/{encounter_id}")
def prediction_history(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT * FROM risk_prediction
        WHERE encounter_id = :eid
        ORDER BY predicted_at DESC
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]


@router.get("/stats/summary")
def prediction_stats(db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT
            COUNT(*)                                       AS total_predictions,
            AVG(fusion_score)                              AS avg_risk_score,
            SUM(CASE WHEN risk_level='HIGH'   THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN risk_level='MEDIUM' THEN 1 ELSE 0 END) AS medium_risk_count,
            SUM(CASE WHEN risk_level='LOW'    THEN 1 ELSE 0 END) AS low_risk_count
        FROM risk_prediction
    """)).fetchone()
    return dict(row._mapping)