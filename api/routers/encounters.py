"""
api/routers/encounters.py
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.database import get_db

router = APIRouter()

@router.get("/")
def list_encounters(limit: int = 30, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            e.encounter_id,
            e.patient_id,
            e.admit_time::date      AS admit_date,
            e.discharge_time::date  AS discharge_date,
            e.admission_type,
            e.discharge_location,
            EXTRACT(EPOCH FROM (
                e.discharge_time - e.admit_time
            ))/86400                AS los_days
        FROM encounter e
        WHERE e.discharge_time IS NOT NULL
        ORDER BY RANDOM()
        LIMIT :limit
    """), {"limit": limit}).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{encounter_id}")
def get_encounter(encounter_id: int, db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT
            e.*,
            p.gender,
            p.anchor_age,
            EXTRACT(EPOCH FROM (
                e.discharge_time - e.admit_time
            ))/86400 AS los_days
        FROM encounter e
        JOIN patient p ON p.patient_id = e.patient_id
        WHERE e.encounter_id = :eid
    """), {"eid": encounter_id}).fetchone()
    if not row:
        raise HTTPException(404, "Encounter not found")
    return dict(row._mapping)

@router.get("/{encounter_id}/diagnoses")
def get_diagnoses(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT icd_code, icd_version, description, seq_num
        FROM diagnosis
        WHERE encounter_id = :eid
        ORDER BY seq_num
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{encounter_id}/vitals")
def get_vitals(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT vital_type, value, unit, recorded_at
        FROM vital_sign
        WHERE encounter_id = :eid
        ORDER BY recorded_at
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{encounter_id}/medications")
def get_medications(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT DISTINCT drug_name, drug_type, route
        FROM medication
        WHERE encounter_id = :eid
        ORDER BY drug_name
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]