"""
api/routers/patients.py
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.database import get_db

router = APIRouter()

@router.get("/")
def list_patients(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            p.patient_id,
            p.gender,
            p.anchor_age,
            p.anchor_year,
            p.dod,
            COUNT(e.encounter_id) AS total_encounters
        FROM patient p
        LEFT JOIN encounter e ON e.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY total_encounters DESC
        LIMIT :limit
    """), {"limit": limit}).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT * FROM patient WHERE patient_id = :pid
    """), {"pid": patient_id}).fetchone()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "Patient not found")
    return dict(row._mapping)

@router.get("/{patient_id}/encounters")
def get_patient_encounters(patient_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            e.encounter_id,
            e.admit_time,
            e.discharge_time,
            e.admission_type,
            e.discharge_location,
            e.hospital_expire_flag,
            EXTRACT(EPOCH FROM (
                e.discharge_time - e.admit_time
            ))/86400 AS los_days
        FROM encounter e
        WHERE e.patient_id = :pid
        ORDER BY e.admit_time DESC
    """), {"pid": patient_id}).fetchall()
    return [dict(r._mapping) for r in rows]