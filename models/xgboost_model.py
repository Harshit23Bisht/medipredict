"""
models/xgboost_model.py
-----------------------
Trains XGBoost on real MIMIC-IV data.
Features come from encounter_features view.
Label comes from readmission_label view.
Run: python models/xgboost_model.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve
)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from dotenv import load_dotenv
import shap

load_dotenv()

DB_URL = os.getenv("POSTGRES_URL",
         os.getenv("DATABASE_URL",
         "postgresql://postgres:hb23@localhost:5432/medipredict"))
engine = create_engine(DB_URL)

os.makedirs('data/models',  exist_ok=True)
os.makedirs('eda_plots',    exist_ok=True)

# ── Feature columns ───────────────────────────────────────────
FEATURE_COLS = [
    'age_at_admission',
    'gender',
    'length_of_stay',
    'num_diagnoses',
    'num_medications',
    'avg_hr',
    'max_bp_sys',
    'avg_temp',
    'max_creatinine',
    'max_wbc',
    'num_prior_admissions',
]

# ── Load Data ─────────────────────────────────────────────────
def load_data():
    print("Loading data from PostgreSQL...")
    query = """
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
        ORDER BY ef.admit_date ASC;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    print(f"Total encounters loaded : {len(df):,}")
    print(f"Readmission rate        : {df['label'].mean():.2%}")
    print(f"Date range              : {df['admit_date'].min()} "
          f"→ {df['admit_date'].max()}")
    return df

# ── Split ─────────────────────────────────────────────────────
def split_data(df):
    """Chronological 70/15/15 split — train/val/test"""
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(f"\nTrain : {len(train):,} "
          f"({train['admit_date'].min()} → {train['admit_date'].max()})")
    print(f"Val   : {len(val):,} "
          f"({val['admit_date'].min()} → {val['admit_date'].max()})")
    print(f"Test  : {len(test):,} "
          f"({test['admit_date'].min()} → {test['admit_date'].max()})")

    return train, val, test

# ── Train ─────────────────────────────────────────────────────
def train_model(train, val):
    print("\nTraining XGBoost...")

    X_train = train[FEATURE_COLS].fillna(0).astype(float)
    y_train = train['label'].astype(int)
    X_val   = val[FEATURE_COLS].fillna(0).astype(float)
    y_val   = val['label'].astype(int)

    # Handle class imbalance
    pos   = y_train.sum()
    neg   = len(y_train) - pos
    scale = neg / pos
    print(f"  Class imbalance ratio (scale_pos_weight): {scale:.2f}")

    model = XGBClassifier(
        n_estimators       = 500,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale,
        eval_metric        = 'auc',
        early_stopping_rounds = 20,
        random_state       = 42,
        n_jobs             = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model, X_train, y_train, X_val, y_val

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, X_test, y_test, X_train, y_train):
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Scores
    train_prob = model.predict_proba(X_train)[:,1]
    test_prob  = model.predict_proba(X_test)[:,1]
    test_pred  = model.predict(X_test)

    train_auc  = roc_auc_score(y_train, train_prob)
    test_auc   = roc_auc_score(y_test,  test_prob)
    test_prauc = average_precision_score(y_test, test_prob)

    print(f"\nTrain ROC-AUC  : {train_auc:.4f}")
    print(f"Test  ROC-AUC  : {test_auc:.4f}")
    print(f"Test  PR-AUC   : {test_prauc:.4f}")
    print(f"Overfit gap    : {train_auc - test_auc:.4f}",
          "✅" if train_auc - test_auc < 0.05 else "⚠️  check overfitting")

    print("\nClassification Report:")
    print(classification_report(
        y_test, test_pred,
        target_names=['Not Readmitted','Readmitted']
    ))

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        print(f"  {feat:<25} {imp:.4f}")

    return test_auc, test_prauc, test_prob, test_pred

# ── Plots ─────────────────────────────────────────────────────
def save_plots(model, X_train, y_train,
               X_test, y_test, test_prob,
               test_pred, test_auc):
    print("\nSaving plots to eda_plots/...")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2.5,
             label=f'XGBoost (AUC={test_auc:.4f})')
    plt.plot([0,1],[0,1],'--', color='gray',
             label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — XGBoost on MIMIC-IV',
              fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('eda_plots/roc_curve.png', dpi=150)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(
        cm, display_labels=['Not Readmitted','Readmitted']
    ).plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix — XGBoost',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/confusion_matrix.png', dpi=150)
    plt.close()

    # 3. Feature Im