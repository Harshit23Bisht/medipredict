"""
api/database.py
---------------
Database connections for PostgreSQL.
MongoDB added tomorrow when network is available.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "POSTGRES_URL",
    os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:hb23@localhost:5432/medipredict"
    )
)

engine       = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()