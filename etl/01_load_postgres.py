"""
ETL Step 1 — Load MIMIC-IV into PostgreSQL
------------------------------------------
Handles .csv and .csv.gz files automatically.
Column names verified against actual MIMIC-IV files.
Run: python etl/01_load_postgres.py
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL  = os.getenv("POSTGRES_URL",
          os.getenv("DATABASE_URL",
          "postgresql://postgres:hb23@localhost:5432/medipredict"))
engine  = create_engine(DB_URL)
RAW     = os.getenv("MIMIC_RAW_PATH", "data/raw")

# ── Helpers ───────────────────────────────────────────────────
def find_file(folder, name):
    for ext in ['.csv.gz', '.csv']:
        path = os.path.join(RAW, folder, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {name}(.csv/.csv.gz) in {RAW}/{folder}/")

def chunk_insert(df, table, chunksize=5000):
    total = len(df)
    for i in range(0, total, chunksize):
        df.iloc[i:i+chunksize].to_sql(
            table, engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        print(f"  {table}: {min(i+chunksize,total):,}/{total:,}",
              end='\r')
    print(f"  {table}: {total:,} rows loaded ✅")

# ── 1. Patients ───────────────────────────────────────────────
def load_patients():
    print("\n[1/7] Loading patients...")
    df = pd.read_csv(find_file('hosp', 'patients'))
    df = df.rename(columns={'subject_id': 'patient_id'})
    df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
    cols = ['patient_id','gender','anchor_age','anchor_year','dod']
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'patient')

# ── 2. Encounters ─────────────────────────────────────────────
def load_encounters():
    print("\n[2/7] Loading encounters...")
    df = pd.read_csv(find_file('hosp', 'admissions'))
    df = df.rename(columns={
        'subject_id':   'patient_id',
        'hadm_id':      'encounter_id',
        'admittime':    'admit_time',
        'dischtime':    'discharge_time',
    })
    df['admit_time']     = pd.to_datetime(
        df['admit_time'],     errors='coerce')
    df['discharge_time'] = pd.to_datetime(
        df['discharge_time'], errors='coerce')
    cols = [
        'encounter_id','patient_id','admit_time',
        'discharge_time','admission_type',
        'admission_location','discharge_location',
        'insurance','marital_status','race',
        'hospital_expire_flag'
    ]
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'encounter')

# ── 3. ICU Stays ──────────────────────────────────────────────
def load_icu_stays():
    print("\n[3/7] Loading ICU stays...")
    df = pd.read_csv(find_file('icu', 'icustays'))
    df = df.rename(columns={
        'subject_id': 'patient_id',
        'hadm_id':    'encounter_id',
        'stay_id':    'icu_stay_id',
        'intime':     'icu_intime',
        'outtime':    'icu_outtime',
        'los':        'los_hours',
    })
    df['icu_intime']  = pd.to_datetime(
        df['icu_intime'],  errors='coerce')
    df['icu_outtime'] = pd.to_datetime(
        df['icu_outtime'], errors='coerce')
    cols = [
        'icu_stay_id','encounter_id','patient_id',
        'first_careunit','last_careunit',
        'icu_intime','icu_outtime','los_hours'
    ]
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'icu_stay')

# ── 4. Diagnoses ──────────────────────────────────────────────
def load_diagnoses():
    print("\n[4/7] Loading diagnoses...")
    df   = pd.read_csv(find_file('hosp', 'diagnoses_icd'))
    desc = pd.read_csv(find_file('hosp', 'd_icd_diagnoses'))
    desc = desc.rename(columns={'long_title': 'description'})
    df   = df.merge(
        desc[['icd_code','icd_version','description']],
        on=['icd_code','icd_version'], how='left'
    )
    df = df.rename(columns={'hadm_id': 'encounter_id'})
    df['description'] = df['description'].fillna('Unknown')
    cols = [
        'encounter_id','icd_code','icd_version',
        'description','seq_num'
    ]
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'diagnosis')

# ── 5. Medications ────────────────────────────────────────────
def load_medications():
    print("\n[5/7] Loading medications...")
    df = pd.read_csv(find_file('hosp', 'prescriptions'))
    df = df.rename(columns={
        'hadm_id':   'encounter_id',
        'drug':      'drug_name',
        'starttime': 'start_time',
        'stoptime':  'stop_time',
    })
    df['start_time'] = pd.to_datetime(
        df['start_time'], errors='coerce')
    df['stop_time']  = pd.to_datetime(
        df['stop_time'],  errors='coerce')
    df['drug_name']  = df['drug_name'].fillna('Unknown')
    # Drop rows with no encounter
    df = df.dropna(subset=['encounter_id'])
    df['encounter_id'] = df['encounter_id'].astype(int)
    cols = [
        'encounter_id','drug_name','drug_type',
        'route','start_time','stop_time'
    ]
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'medication')

# ── 6. Vitals ─────────────────────────────────────────────────
def load_vitals():
    print("\n[6/7] Loading vitals (largest file ~3GB)...")
    print("  This will take 15-30 minutes...")

    VITAL_ITEMS = {
        220045: ('heart_rate',       'bpm'),
        220179: ('bp_systolic',      'mmHg'),
        220180: ('bp_diastolic',     'mmHg'),
        223761: ('temperature_f',    'F'),
        220210: ('respiratory_rate', 'br/min'),
        220277: ('spo2',             '%'),
        220739: ('gcs_eye',          'points'),
        223900: ('gcs_verbal',       'points'),
        223901: ('gcs_motor',        'points'),
    }

    chunks = pd.read_csv(
        find_file('icu', 'chartevents'),
        usecols=[
            'subject_id','hadm_id','stay_id',
            'charttime','itemid','valuenum'
        ],
        dtype={
            'hadm_id':  'Int64',
            'stay_id':  'Int64',
            'itemid':   'int32',
        },
        chunksize=500_000
    )

    total = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk[
            chunk['itemid'].isin(VITAL_ITEMS)
        ].copy()
        if chunk.empty:
            continue

        chunk['vital_type']   = chunk['itemid'].map(
            lambda x: VITAL_ITEMS[x][0])
        chunk['unit']         = chunk['itemid'].map(
            lambda x: VITAL_ITEMS[x][1])
        chunk['encounter_id'] = chunk['hadm_id']
        chunk['icu_stay_id']  = chunk['stay_id']
        chunk['value']        = pd.to_numeric(
            chunk['valuenum'], errors='coerce')
        chunk['recorded_at']  = pd.to_datetime(
            chunk['charttime'], errors='coerce')

        out = chunk[[
            'encounter_id','icu_stay_id',
            'vital_type','value','unit','recorded_at'
        ]].dropna(subset=['value','encounter_id'])

        if not out.empty:
            out.to_sql(
                'vital_sign', engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=5000
            )
            total += len(out)

        print(f"  Chunk {i+1} done | "
              f"Total vitals: {total:,}", end='\r')

    print(f"\n  Vitals loaded: {total:,} ✅")

# ── 7. Labs ───────────────────────────────────────────────────
def load_labs():
    print("\n[7/7] Loading labs (2.4GB file)...")
    print("  This will take 10-20 minutes...")

    KEY_LABS = {
        50912: 'Creatinine',
        51006: 'Blood Urea Nitrogen',
        51301: 'White Blood Cells',
        51222: 'Hemoglobin',
        50983: 'Sodium',
        50971: 'Potassium',
        50931: 'Glucose',
        51003: 'Troponin T',
        50813: 'Lactate',
        50862: 'Albumin',
        51265: 'Platelet Count',
        50882: 'Bicarbonate',
        50893: 'Calcium Total',
        51144: 'Bands',
        50902: 'Chloride',
    }

    chunks = pd.read_csv(
        find_file('hosp', 'labevents'),
        usecols=[
            'labevent_id','subject_id','hadm_id',
            'itemid','charttime','value',
            'valuenum','valueuom','flag'
        ],
        dtype={
            'hadm_id':    'Int64',
            'subject_id': 'Int64',
            'itemid':     'int32',
        },
        chunksize=500_000
    )

    total = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk[
            chunk['itemid'].isin(KEY_LABS)
        ].copy()
        # Drop rows with no hospital admission
        chunk = chunk.dropna(subset=['hadm_id'])
        if chunk.empty:
            continue

        chunk['lab_name']     = chunk['itemid'].map(KEY_LABS)
        chunk['encounter_id'] = chunk['hadm_id'].astype(int)
        chunk['patient_id']   = chunk['subject_id'].astype(int)
        chunk['valuenum']     = pd.to_numeric(
            chunk['valuenum'], errors='coerce')
        chunk['chart_time']   = pd.to_datetime(
            chunk['charttime'], errors='coerce')

        out = chunk[[
            'encounter_id','patient_id','lab_name',
            'itemid','value','valuenum','valueuom',
            'flag','chart_time'
        ]]

        if not out.empty:
            out.to_sql(
                'lab_result', engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=5000
            )
            total += len(out)

        print(f"  Chunk {i+1} done | "
              f"Total labs: {total:,}", end='\r')

    print(f"\n  Labs loaded: {total:,} ✅")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("MediPredict ETL — MIMIC-IV → PostgreSQL")
    print("=" * 55)

    # print("\nClearing existing data...")
    # with engine.connect() as conn:
    #     conn.execute(text("""
    #         TRUNCATE TABLE
    #             risk_prediction,
    #             vital_sign,
    #             lab_result,
    #             medication,
    #             diagnosis,
    #             icu_stay,
    #             encounter,
    #             patient
    #         CASCADE;
    #     """))
    #     conn.commit()
    # print("  Cleared ✅")

    # load_patients()
    # load_encounters()
    # load_icu_stays()
    # load_diagnoses()
    load_medications()
    # # load_vitals()
    # load_labs()

    print("\n" + "=" * 55)
    print("✅ ETL Complete!")
    print("=" * 55)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT 'patients'    AS tbl, COUNT(*) AS n
            FROM patient
            UNION ALL
            SELECT 'encounters',  COUNT(*) FROM encounter
            UNION ALL
            SELECT 'icu_stays',   COUNT(*) FROM icu_stay
            UNION ALL
            SELECT 'diagnoses',   COUNT(*) FROM diagnosis
            UNION ALL
            SELECT 'medications', COUNT(*) FROM medication
            UNION ALL
            SELECT 'vitals',      COUNT(*) FROM vital_sign
            UNION ALL
            SELECT 'labs',        COUNT(*) FROM lab_result
        """))
        print(f"\n{'Table':<15} {'Rows':>12}")
        print("-" * 28)
        for r in rows:
            print(f"  {r[0]:<13} {r[1]:>12,}")