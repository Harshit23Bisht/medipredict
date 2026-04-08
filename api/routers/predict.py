"""
api/routers/predict.py
----------------------
Prediction endpoints.
Uses XGBoost (always) + LSTM (if available).
Fusion combines both scores.
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.database import get_db
import api.main as app_state

router = APIRouter()

FEATURE_COLS = [
    'age_at_admission', 'gender', 'length_of_stay',
    'num_diagnoses', 'num_medications', 'avg_hr',
    'max_bp_sys', 'avg_temp', 'max_creatinine',
    'max_wbc', 'num_prior_admissions',
]

def get_xgb_score(encounter_id: int, db: Session) -> dict:
    """Get XGBoost risk score from encounter_features view."""
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
        raise HTTPException(
            404, f"No features found for encounter {encounter_id}")

    features = dict(row._mapping)
    xgb_data  = app_state.models.get("xgboost")
    if not xgb_data:
        raise HTTPException(503, "XGBoost model not loaded")

    model = xgb_data["model"]
    X     = pd.DataFrame([features])[FEATURE_COLS].fillna(0).astype(float)
    score = float(model.predict_proba(X)[0, 1])
    return {"score": score, "features": features}

def get_lstm_score(encounter_id: int, db: Session) -> float | None:
    """Get LSTM risk score from vital sign sequences."""
    lstm_data = app_state.models.get("lstm")
    if not lstm_data:
        return None

    # Get hourly vitals sequence
    rows = db.execute(text("""
        SELECT
            vital_type,
            AVG(value) AS value,
            DATE_TRUNC('hour', recorded_at) AS hour
        FROM vital_sign
        WHERE encounter_id = :eid
        AND vital_type IN (
            'heart_rate','bp_systolic','bp_diastolic',
            'temperature_f','respiratory_rate','spo2'
        )
        GROUP BY vital_type, DATE_TRUNC('hour', recorded_at)
        ORDER BY hour
    """), {"eid": encounter_id}).fetchall()

    if not rows:
        return None

    try:
        model     = lstm_data["model"]
        seq_len   = lstm_data.get("seq_length", 24)
        vitals    = ['heart_rate','bp_systolic','bp_diastolic',
                     'temperature_f','respiratory_rate','spo2']
        scaler    = lstm_data.get("scaler")

        # Build pivot
        df = pd.DataFrame([dict(r._mapping) for r in rows])
        df = df.pivot_table(
            index='hour', columns='vital_type',
            values='value', aggfunc='mean'
        ).reindex(columns=vitals).fillna(method='ffill').fillna(0)

        # Pad / trim to seq_len
        arr = df.values
        if len(arr) < seq_len:
            pad = np.zeros((seq_len - len(arr), len(vitals)))
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-seq_len:]

        if scaler:
            arr = scaler.transform(arr)

        arr = arr.reshape(1, seq_len, len(vitals))
        score = float(model.predict(arr)[0][0])
        return score
    except Exception as e:
        print(f"LSTM scoring error: {e}")
        return None

def risk_level(score: float) -> str:
    if score < 0.30:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    return "HIGH"

@router.post("/{encounter_id}")
def predict(encounter_id: int, db: Session = Depends(get_db)):
    # XGBoost score
    xgb_result  = get_xgb_score(encounter_id, db)
    xgb_score   = xgb_result["score"]
    features    = xgb_result["features"]

    # LSTM score
    lstm_score  = get_lstm_score(encounter_id, db)

    # Fusion
    if lstm_score is not None:
        fusion_score = 0.6 * xgb_score + 0.4 * lstm_score
    else:
        fusion_score = xgb_score

    # Save to DB
    db.execute(text("""
        INSERT INTO risk_prediction
            (encounter_id, xgb_score, lstm_score,
             fusion_score, risk_level, model_version)
        VALUES
            (:eid, :xgb, :lstm, :fusion, :level, :ver)
    """), {
        "eid":    encounter_id,
        "xgb":    round(xgb_score,   4),
        "lstm":   round(lstm_score,  4) if lstm_score else None,
        "fusion": round(fusion_score,4),
        "level":  risk_level(fusion_score),
        "ver":    "v2_mimic",
    })
    db.commit()

    return {
        "encounter_id":  encounter_id,
        "xgb_score":     round(xgb_score,   4),
        "lstm_score":    round(lstm_score,  4) if lstm_score else None,
        "fusion_score":  round(fusion_score,4),
        "risk_score":    round(fusion_score,4),
        "risk_level":    risk_level(fusion_score),
        "model_version": "v2_mimic",
        "features_used": features,
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
            COUNT(*)                          AS total_predictions,
            AVG(fusion_score)                 AS avg_risk_score,
            SUM(CASE WHEN risk_level='HIGH'
                THEN 1 ELSE 0 END)            AS high_risk_count,
            SUM(CASE WHEN risk_level='MEDIUM'
                THEN 1 ELSE 0 END)            AS medium_risk_count,
            SUM(CASE WHEN risk_level='LOW'
                THEN 1 ELSE 0 END)            AS low_risk_count
        FROM risk_prediction
    """)).fetchone()
    return dict(row._mapping)