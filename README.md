# MediPredict — Multimodal Hospital Readmission Risk Prediction System

## Overview

MediPredict is an end-to-end multimodal healthcare AI platform designed to predict 30-day hospital readmission risk using structured Electronic Health Records (EHR), ICU time-series vitals, and chest X-ray imaging.

The system combines:

* XGBoost for structured clinical tabular data
* Bidirectional LSTM for temporal ICU vital-sign sequences
* CNN-based chest X-ray analysis using transfer learning
* Dynamic multimodal weighted fusion for robust prediction under missing modalities

The project integrates database engineering, ETL pipelines, machine learning, deep learning, backend APIs, and frontend visualization into a unified healthcare intelligence system.

---

# Key Features

* Multimodal clinical AI pipeline
* PostgreSQL-based MIMIC-style healthcare database
* SQL feature engineering workflows
* ICU time-series modeling using BiLSTM
* Chest X-ray analysis using CNNs
* Dynamic modality-aware fusion engine
* FastAPI backend APIs
* Frontend dashboard for inference and visualization
* End-to-end training and deployment pipeline

---

# System Architecture

```text
Raw Clinical Data
        ↓
PostgreSQL Database
        ↓
ETL + SQL Feature Engineering
        ↓
ML / DL Models
 ├── XGBoost (Tabular EHR)
 ├── BiLSTM (Temporal Vitals)
 └── CNN (Chest X-rays)
        ↓
Dynamic Weighted Fusion
        ↓
FastAPI Backend
        ↓
Frontend Dashboard
```

---

# Tech Stack

## Database & Data Engineering

* PostgreSQL
* SQL
* SQLAlchemy
* Pandas
* NumPy

## Machine Learning & Deep Learning

* XGBoost
* PyTorch
* Scikit-learn
* Bidirectional LSTM
* ResNet-based CNN

## Backend & APIs

* FastAPI
* Uvicorn
* REST APIs

## Frontend

* HTML
* CSS
* JavaScript

## DevOps & Tooling

* Git
* GitHub
* Docker
* Python Virtual Environments

---

# Database Design

The database schema is inspired by the MIMIC-IV clinical database structure.

## Core Tables

* patient
* encounter
* diagnosis
* medication
* lab_result
* vital_sign

## Derived Views

### encounter_features

Aggregated encounter-level features:

* age_at_admission
* length_of_stay
* num_diagnoses
* num_medications
* max_creatinine
* max_wbc
* avg_hr
* max_bp_sys
* avg_temp
* num_prior_admissions

### readmission_label

Generated using SQL window functions to identify future readmissions.

---

# Machine Learning Pipeline

## 1. XGBoost — Structured Clinical Intelligence

Used for:

* demographics
* medications
* diagnoses
* lab summaries
* encounter statistics

### Performance

* ROC-AUC: ~0.75
* PR-AUC: ~0.14

### Why XGBoost?

* Excellent performance on tabular clinical data
* Handles missing values effectively
* Provides interpretable feature importance
* Clinically explainable outputs

---

## 2. Bidirectional LSTM — Temporal ICU Modeling

Processes sequential ICU vital-sign data:

* heart rate
* blood pressure
* respiratory rate
* oxygen saturation
* temperature

### Performance

* ROC-AUC: ~0.90

### Why LSTM?

Captures temporal physiological trends that static models cannot learn.

---

## 3. CNN — Chest X-ray Analysis

CNN-based imaging pipeline using transfer learning with a ResNet backbone.

### Performance

* ROC-AUC: ~0.92

### Purpose

Extracts visual pathology indicators from chest X-rays for enhanced multimodal risk prediction.

---

# Dynamic Multimodal Fusion

The fusion engine combines predictions from:

* XGBoost
* BiLSTM
* CNN

using weighted modality-aware fusion.

## Important Capability

The system dynamically adapts when modalities are unavailable.

Example:

* If chest X-rays are unavailable:

  * prediction is generated using only XGBoost + LSTM
* If ICU sequences are missing:

  * fusion uses tabular + imaging models

This makes the system robust for real-world clinical deployment where patient data is often incomplete.

---

# Frontend Dashboard

The frontend dashboard supports:

* encounter-based prediction
* image upload for chest X-rays
* multimodal fusion visualization
* real-time inference
* risk-level display

The frontend communicates with FastAPI backend services through REST APIs.

---

# API Endpoints

## Prediction Endpoint

```http
POST /predict
```

Returns:

```json
{
  "encounter_id": 24085783,
  "xgb_score": 0.49,
  "lstm_score": 0.48,
  "cnn_score": 0.92,
  "fusion_score": 0.81,
  "risk_level": "HIGH"
}
```

---

# Project Structure

```text
MediPredict/
│
├── api/                        # FastAPI backend services
│   ├── routers/                # API route handlers
│   │   ├── predict.py          # Multimodal prediction endpoints
│   │   ├── encounters.py
│   │   ├── labs.py
│   │   └── patients.py
│   ├── database.py             # PostgreSQL connection configuration
│   └── main.py                 # FastAPI application entrypoint
│
├── models/                     # Machine learning & deep learning models
│   ├── xgboost_model.py        # Structured EHR prediction
│   ├── lstm_model.py           # ICU temporal sequence modeling
│   ├── cnn_model.py            # Chest X-ray CNN pipeline
│   └── fusion.py               # Dynamic multimodal fusion engine
│
├── data/
│   ├── models/                 # Saved trained model artifacts
│   │   ├── xgboost_mimic_v1.pkl
│   │   ├── lstm_mimic_v1.pkl
│   │   └── cnn_v1.pth
│   ├── processed/              # Processed feature-engineered datasets
│   └── raw/                    # Raw datasets and chest X-ray images
│
├── etl/                        # ETL and feature engineering pipelines
│   ├── 01_load_postgres.py
│   ├── 02_load_mongo.py
│   └── 03_feature_eng.py
│
├── frontend/                   # Frontend dashboard UI
│   └── index.html
│
├── db/                         # Database schema and SQL scripts
├── eda_plots/                  # Exploratory data analysis outputs
├── notebooks/                  # Research and experimentation notebooks
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Multi-container orchestration
├── .env.example                # Environment variable template
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <repository-url>
cd MediPredict
```

## 2. Create Virtual Environment

### Windows (Git Bash)

```bash
python -m venv venv
source venv/Scripts/activate
```

### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Database Setup

## PostgreSQL

Create database:

```sql
CREATE DATABASE medipredict;
```

Configure environment variables:

```env
POSTGRES_URL=postgresql://postgres:password@localhost:5432/medipredict
```

---

# Running the Backend

```bash
uvicorn api.main:app --reload --port 8000
```

API Docs:

```text
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Open new terminal:

```bash
cd frontend
python -m http.server 5500
```

Frontend URL:

```text
http://127.0.0.1:5500
```

---

# Running Models

## XGBoost

```bash
python models/xgboost_model.py
```

## LSTM

```bash
python models/lstm_model.py
```

## Fusion Pipeline

```bash
python models/fusion.py --encounter_id 24085783 --image_path data/raw/CNN/pneumonia/person1_virus_6.jpeg
```

---

# Example Output

```text
encounter_id : 24085783
xgb_score    : 0.49
lstm_score   : 0.48
cnn_score    : 0.92
fusion_score : 0.81
risk_level   : HIGH
```

---

# Research & Innovation Highlights

* Multimodal healthcare AI system
* Dynamic missing-modality fusion
* Time-series ICU modeling
* Explainable clinical feature engineering
* Real-world MIMIC-style healthcare schema
* Full-stack AI deployment pipeline

---

# Future Improvements

* Transformer-based temporal fusion
* Clinical note NLP integration
* Federated healthcare learning
* Explainable multimodal attention maps
* Real-time streaming ICU monitoring
* Deployment on cloud infrastructure

---

# Contributors

* Harshit Bisht

---

# License

This project is intended for educational and research purposes.
