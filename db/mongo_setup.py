"""
Sets up MongoDB collections with indexes
Run: python db/mongo_setup.py

⚠️ NOTE FOR TEAM:
This file defines the REQUIRED MongoDB structure for the project.
Do NOT change critical field names used for integration with PostgreSQL.
You may ADD fields/indexes, but do not break existing ones.
"""

from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import os

load_dotenv()

# 🔹 Connect to MongoDB using environment variables
client = MongoClient(os.getenv("MONGO_URL"))
db     = client[os.getenv("MONGO_DB")]

# =========================================================
# 🖼 IMAGES COLLECTION
# =========================================================
# Stores image metadata for CNN models
# Example document:
# {
#   "postgres_encounter_id": "123",   ← MUST MATCH PostgreSQL encounter_id
#   "image_path": "...",
#   "modality": "X-ray",
#   "study_date": "YYYY-MM-DD",
#   ...
# }

# ⚠️ CRITICAL: used to link MongoDB images with PostgreSQL data
db.images.create_index([("postgres_encounter_id", ASCENDING)])

# Used for filtering images by type (X-ray, MRI, etc.)
db.images.create_index([("modality", ASCENDING)])

# Used for time-based queries (important for temporal models)
db.images.create_index([("study_date", ASCENDING)])


# =========================================================
# 📝 CLINICAL NOTES COLLECTION
# =========================================================
# Stores text notes (for NLP / BERT models)
# Example document:
# {
#   "postgres_encounter_id": "123",   ← MUST MATCH PostgreSQL
#   "note_type": "discharge",
#   "text": "...",
#   ...
# }

# ⚠️ CRITICAL: linking field with PostgreSQL
db.clinical_notes.create_index([("postgres_encounter_id", ASCENDING)])

# Used to filter note types (discharge, radiology, etc.)
db.clinical_notes.create_index([("note_type", ASCENDING)])


# =========================================================
# 🧠 EMBEDDINGS COLLECTION
# =========================================================
# Stores vector embeddings (from CNN / BERT)
# Example document:
# {
#   "encounter_id": "123",   ← MUST MATCH PostgreSQL encounter_id
#   "embedding_type": "cnn" or "bert",
#   "vector": [...],
# }

# ⚠️ CRITICAL: linking embeddings to encounter
db.embeddings.create_index([("encounter_id", ASCENDING)])

# Used to differentiate embedding types (cnn, bert, etc.)
db.embeddings.create_index([("embedding_type", ASCENDING)])


# =========================================================
# ✅ DONE
# =========================================================
print("✅ MongoDB collections and indexes created!")
print(f"   Collections: {db.list_collection_names()}")