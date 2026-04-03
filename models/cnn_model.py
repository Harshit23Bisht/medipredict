import torch
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import numpy as np

# Import your existing code
from src.model import get_model
from src.dataset import sakaguchi_transform, bipolar_fuzzy_enhancement
import config

# Mongo + Postgres
from pymongo import MongoClient
import psycopg2


# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Apply Sakaguchi + Fuzzy
    image = sakaguchi_transform(image)
    image = bipolar_fuzzy_enhancement(image)
    image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = transform(image).unsqueeze(0)
    return image


# ===============================
# Load CNN Model
# ===============================
def load_cnn_model():
    model = get_model()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    return model


# ===============================
# MongoDB Connection
# ===============================
def get_mongo_db():
    client = MongoClient("mongodb://mongo:27017/")
    db = client["medipredict"]
    return db


# ===============================
# PostgreSQL Connection
# ===============================
def get_postgres_connection():
    conn = psycopg2.connect(
        host="postgres",
        database="medipredict",
        user="postgres",
        password="postgres"
    )
    return conn


# ===============================
# Main CNN Pipeline
# ===============================
def run_cnn_model(image_path, encounter_id, patient_id):

    print("[CNN] Loading model...")
    model = load_cnn_model()

    print("[CNN] Preprocessing image...")
    image_tensor = preprocess_image(image_path).to(config.DEVICE)

    print("[CNN] Running inference...")
    with torch.no_grad():
        # Extract embedding vector (2048 features)
        features = model.backbone(image_tensor)
        embedding_vector = features.squeeze().cpu().numpy().tolist()

        # Predict risk score
        risk_score = model.predict_proba(image_tensor).item()

    print("[CNN] Risk Score:", risk_score)

    # ---------------- MongoDB ----------------
    print("[CNN] Storing image metadata in MongoDB...")
    mongo_db = get_mongo_db()

    image_doc = {
        "postgres_encounter_id": encounter_id,
        "image_path": image_path,
        "modality": "X-ray",
        "study_date": datetime.utcnow()
    }

    image_result = mongo_db.images.insert_one(image_doc)
    mongo_id = str(image_result.inserted_id)

    # Store embedding
    embedding_doc = {
        "encounter_id": encounter_id,
        "embedding_type": "cnn",
        "vector": embedding_vector
    }

    mongo_db.embeddings.insert_one(embedding_doc)

    # ---------------- PostgreSQL ----------------
    print("[CNN] Storing results in PostgreSQL...")
    conn = get_postgres_connection()
    cur = conn.cursor()

    # Insert imaging study
    cur.execute("""
        INSERT INTO imaging_study (
            encounter_id,
            patient_id,
            modality,
            body_part,
            mongo_id,
            file_name,
            study_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        encounter_id,
        patient_id,
        "X-ray",
        "Chest",
        mongo_id,
        image_path.split("/")[-1],
        datetime.utcnow()
    ))

    # Insert CNN risk score
    cur.execute("""
        INSERT INTO risk_prediction (
            encounter_id,
            cnn_score,
            model_version,
            predicted_at
        )
        VALUES (%s, %s, %s, %s)
    """, (
        encounter_id,
        risk_score,
        config.MODEL_VERSION,
        datetime.utcnow()
    ))

    conn.commit()
    cur.close()
    conn.close()

    print("[CNN] Data stored successfully.")

    return {
        "encounter_id": encounter_id,
        "cnn_score": risk_score,
        "model_version": config.MODEL_VERSION
    }
