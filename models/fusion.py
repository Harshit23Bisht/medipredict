"""
models/fusion.py
----------------
Loads trained XGBoost + LSTM (+ optional CNN) models and produces
a fused readmission risk score for a single encounter or a batch.

Usage (standalone test):
    python models/fusion.py --encounter_id 12345

Usage (from API):
    from models.fusion import FusionPredictor
    predictor = FusionPredictor()
    result    = predictor.predict(encounter_id=12345)
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "POSTGRES_URL",
    os.getenv("DATABASE_URL",
              "postgresql://postgres:hb23@localhost:5432/medipredict"),
)

# ── Fusion weights ────────────────────────────────────────────
# Adjust once CNN is integrated by your teammate
WEIGHTS = {
    "xgb":  0.60,
    "lstm": 0.40,
    "cnn":  0.00,   # activated when CNN model is present
}

# ── Risk thresholds ───────────────────────────────────────────
def risk_level(score: float) -> str:
    if score < 0.30:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    return "HIGH"


# ─────────────────────────────────────────────────────────────
class FusionPredictor:
    """
    Loads models once at construction, then serves predictions fast.

    Parameters
    ----------
    xgb_path  : path to xgboost_mimic_v1.pkl
    lstm_path : path to lstm_mimic_v1.pkl
    cnn_path  : path to cnn_model.pkl  (optional — skipped if missing)
    """

    XGB_PATH  = "data/models/xgboost_mimic_v1.pkl"
    LSTM_PATH = "data/models/lstm_mimic_v1.pkl"
    CNN_PATH  = "data/models/cnn_model.pkl"

    def __init__(self,
                 xgb_path:  str | None = None,
                 lstm_path: str | None = None,
                 cnn_path:  str | None = None):

        self.engine = create_engine(DB_URL, pool_pre_ping=True)

        # XGBoost ────────────────────────────────────────────
        xgb_path = xgb_path or self.XGB_PATH
        with open(xgb_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.xgb_model    = bundle["model"]
        self.xgb_features = bundle["feature_cols"]
        print(f"  [fusion] XGBoost loaded  ← {xgb_path}")

        # LSTM ───────────────────────────────────────────────
        lstm_path = lstm_path or self.LSTM_PATH
        with open(lstm_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.lstm_model    = bundle["model"]
        self.lstm_scaler   = bundle["scaler"]
        self.lstm_vitals   = bundle["vital_cols"]
        self.seq_length    = bundle["seq_length"]
        print(f"  [fusion] LSTM loaded     ← {lstm_path}")

        # CNN (optional) ─────────────────────────────────────
        cnn_path = cnn_path or self.CNN_PATH
        if os.path.exists(cnn_path):
            with open(cnn_path, "rb") as fh:
                self.cnn_bundle = pickle.load(fh)
            print(f"  [fusion] CNN loaded      ← {cnn_path}")
            self._cnn_available = True
        else:
            self.cnn_bundle     = None
            self._cnn_available = False
            print(f"  [fusion] CNN not found — will use XGB+LSTM only")

        self._set_weights()

    # ─────────────────────────────────────────────────────────
    def _set_weights(self):
        """Recompute weights so they sum to 1.0."""
        w = WEIGHTS.copy()
        if not self._cnn_available:
            # redistribute CNN weight to XGB and LSTM proportionally
            total_without_cnn = w["xgb"] + w["lstm"]
            w["xgb"]  = w["xgb"]  / total_without_cnn
            w["lstm"] = w["lstm"] / total_without_cnn
            w["cnn"]  = 0.0
        self._weights = w

    # ─────────────────────────────────────────────────────────
    # Tabular features for XGBoost
    # ─────────────────────────────────────────────────────────
    def _fetch_tabular(self, encounter_id: int) -> pd.DataFrame:
        query = text("""
            SELECT
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
                ef.num_prior_admissions
            FROM encounter_features ef
            WHERE ef.encounter_id = :eid
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"eid": encounter_id})
        return df

    # ─────────────────────────────────────────────────────────
    # Vital sequences for LSTM
    # ─────────────────────────────────────────────────────────
    def _fetch_vitals(self, encounter_id: int) -> np.ndarray:
        """
        Returns (1, SEQ_LENGTH, N_FEATURES) normalised array.
        Uses vital_sequence table if the encounter is an ICU stay,
        otherwise falls back to recent vital_sign rows.
        """
        # Try vital_sequence first (ICU encounters)
        query = text("""
            SELECT hour_offset, heart_rate, bp_systolic, bp_diastolic,
                   temperature_f, respiratory_rate, spo2
            FROM vital_sequence
            WHERE encounter_id = :eid
            ORDER BY hour_offset
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"eid": encounter_id})

        if df.empty:
            # Fallback: use latest hourly vitals from vital_sign
            query2 = text("""
                SELECT
                    AVG(CASE WHEN vital_type='heart_rate'       THEN value END) AS heart_rate,
                    AVG(CASE WHEN vital_type='bp_systolic'      THEN value END) AS bp_systolic,
                    AVG(CASE WHEN vital_type='bp_diastolic'     THEN value END) AS bp_diastolic,
                    AVG(CASE WHEN vital_type='temperature_f'    THEN value END) AS temperature_f,
                    AVG(CASE WHEN vital_type='respiratory_rate' THEN value END) AS respiratory_rate,
                    AVG(CASE WHEN vital_type='spo2'             THEN value END) AS spo2,
                    DATE_TRUNC('hour', recorded_at) AS hour
                FROM vital_sign
                WHERE encounter_id = :eid
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT :sl
            """)
            with self.engine.connect() as conn:
                df = pd.read_sql(query2, conn,
                                 params={"eid": encounter_id, "sl": self.seq_length})
            df = df[self.lstm_vitals].fillna(0)
        else:
            df = df[self.lstm_vitals].fillna(0)

        arr = df.values.astype(np.float32)

        # Shape to (SEQ_LENGTH, N_FEATURES)
        if len(arr) >= self.seq_length:
            arr = arr[-self.seq_length:]
        else:
            pad = np.zeros((self.seq_length - len(arr), len(self.lstm_vitals)),
                           dtype=np.float32)
            arr = np.vstack([pad, arr])

        # Normalise
        n_feat = len(self.lstm_vitals)
        arr    = self.lstm_scaler.transform(
            arr.reshape(-1, n_feat)
        ).reshape(1, self.seq_length, n_feat)

        return arr

    # ─────────────────────────────────────────────────────────
    # XGBoost score
    # ─────────────────────────────────────────────────────────
    def _xgb_score(self, encounter_id: int) -> float:
        df = self._fetch_tabular(encounter_id)
        if df.empty:
            return 0.5   # neutral fallback
        X = df[self.xgb_features].fillna(0).astype(float)
        return float(self.xgb_model.predict_proba(X)[0, 1])

    # ─────────────────────────────────────────────────────────
    # LSTM score
    # ─────────────────────────────────────────────────────────
    def _lstm_score(self, encounter_id: int) -> float:
        import torch
        arr = self._fetch_vitals(encounter_id)   # (1, T, F)
        self.lstm_model.eval()
        with torch.no_grad():
            score = self.lstm_model(
                torch.FloatTensor(arr)
            ).squeeze().item()
        return float(score)

    # ─────────────────────────────────────────────────────────
    # CNN score  (optional — returns None if not available)
    # ─────────────────────────────────────────────────────────
    def _cnn_score(self, image_array: np.ndarray | None) -> float | None:
        if not self._cnn_available or image_array is None:
            return None
        # Your teammate's model — adapt call signature as needed
        cnn_model = self.cnn_bundle.get("model")
        if cnn_model is None:
            return None
        import torch
        cnn_model.eval()
        with torch.no_grad():
            t     = torch.FloatTensor(image_array).unsqueeze(0)
            score = torch.sigmoid(cnn_model(t)).squeeze().item()
        return float(score)

    # ─────────────────────────────────────────────────────────
    # Public predict
    # ─────────────────────────────────────────────────────────
    def predict(
        self,
        encounter_id: int,
        image_array: np.ndarray | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        encounter_id : int   MIMIC hadm_id
        image_array  : optional numpy array for CNN (from X-ray upload)

        Returns
        -------
        dict with keys:
            encounter_id, xgb_score, lstm_score, cnn_score,
            fusion_score, risk_level, weights_used
        """
        xgb_score  = self._xgb_score(encounter_id)
        lstm_score = self._lstm_score(encounter_id)
        cnn_score  = self._cnn_score(image_array)

        w = self._weights
        if cnn_score is not None and self._cnn_available:
            fusion = (w["xgb"] * xgb_score
                      + w["lstm"] * lstm_score
                      + w["cnn"]  * cnn_score)
        else:
            fusion = (w["xgb"] * xgb_score
                      + w["lstm"] * lstm_score)

        return {
            "encounter_id": encounter_id,
            "xgb_score":    round(xgb_score,  4),
            "lstm_score":   round(lstm_score, 4),
            "cnn_score":    round(cnn_score,  4) if cnn_score is not None else None,
            "fusion_score": round(fusion,     4),
            "risk_level":   risk_level(fusion),
            "weights_used": w,
        }

    # ─────────────────────────────────────────────────────────
    # Batch predict  (for offline evaluation or bulk API calls)
    # ─────────────────────────────────────────────────────────
    def predict_batch(self, encounter_ids: list[int]) -> pd.DataFrame:
        rows = []
        for i, eid in enumerate(encounter_ids):
            try:
                result = self.predict(eid)
                rows.append(result)
            except Exception as exc:
                print(f"  ⚠️  encounter {eid} failed: {exc}")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(encounter_ids)} done")
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediPredict Fusion Test")
    parser.add_argument("--encounter_id", type=int, required=True,
                        help="MIMIC encounter_id (hadm_id) to predict")
    args = parser.parse_args()

    print("=" * 52)
    print("MediPredict — Fusion Predictor Test")
    print("=" * 52)

    predictor = FusionPredictor()
    result    = predictor.predict(args.encounter_id)

    print("\nResult:")
    for k, v in result.items():
        print(f"  {k:<15} : {v}")