# MediPredict Setup

## Prerequisites
- Docker Desktop installed
- Git installed

## Run the project
git clone <repo-url>
cd medipredict
cp .env.example .env
docker-compose up

## Access
- Frontend:  http://localhost:3000
- API:       http://localhost:8001
- API Docs:  http://localhost:8001/docs

## Load data (after Docker is running)
docker-compose exec api python etl/01_load_postgres.py

## Team task assignments
- Person 1: XGBoost model (models/xgboost_model.py)
- Person 2: LSTM model (models/lstm_model.py)
- Person 3: CNN model (models/cnn_model.py)
- Person 4: API + Frontend (api/ and frontend/)
