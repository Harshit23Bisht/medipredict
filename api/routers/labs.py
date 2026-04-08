"""
api/routers/labs.py
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.database import get_db

router = APIRouter()

@router.get("/{encounter_id}")
def get_labs(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            lab_name,
            MAX(valuenum)  AS max_value,
            MIN(valuenum)  AS min_value,
            AVG(valuenum)  AS avg_value,
            MAX(valueuom)  AS unit,
            MAX(flag)      AS flag,
            COUNT(*)       AS num_readings
        FROM lab_result
        WHERE encounter_id = :eid
        AND valuenum IS NOT NULL
        GROUP BY lab_name
        ORDER BY lab_name
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/abnormal/{encounter_id}")
def get_abnormal_labs(encounter_id: int, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            lab_name, valuenum, valueuom,
            flag, chart_time
        FROM lab_result
        WHERE encounter_id = :eid
        AND flag = 'abnormal'
        ORDER BY chart_time DESC
    """), {"eid": encounter_id}).fetchall()
    return [dict(r._mapping) for r in rows]