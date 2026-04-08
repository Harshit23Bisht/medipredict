"""
etl/01_load_postgres.py
-----------------------
Loads a stratified sample of MIMIC-IV into PostgreSQL.
Max 10,000 encounters, with proportional related records.
Run: python etl/01_load_postgres.py
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("POSTGRES_URL",
         os.getenv("DATABASE_URL",
         "postgresql://postgres:hb23@localhost:5432/medipredict"))
engine = create_engine(DB_URL)
RAW    = os.getenv("MIMIC_RAW_PATH", "data/raw")

# ── Config ────────────────────────────────────────────────────
MAX_ENCOUNTERS = 30_000   # hard cap
MAX_PATIENTS   = 5_000    # patients to seed from

# ── Helpers ───────────────────────────────────────────────────
def find_file(folder, name):
    for ext in ['.csv.gz', '.csv']:
        path = os.path.join(RAW, folder, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {name} in {RAW}/{folder}/")

def chunk_insert(df, table, chunksize=2000):
    total = len(df)
    if total == 0:
        print(f"  {table}: 0 rows — skipping")
        return
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

# ── Step 1: Sample patients and encounters ────────────────────
def sample_encounters():
    """
    Load admissions, pick MAX_ENCOUNTERS with good
    readmission representation (stratified by admission type).
    Returns: (patient_ids set, encounter_ids set)
    """
    print("\n[Sampling] Loading admissions...")
    adm = pd.read_csv(find_file('hosp', 'admissions'))
    adm = adm.rename(columns={
        'subject_id': 'patient_id',
        'hadm_id':    'encounter_id',
        'admittime':  'admit_time',
        'dischtime':  'discharge_time',
    })
    adm['admit_time']     = pd.to_datetime(
        adm['admit_time'],     errors='coerce')
    adm['discharge_time'] = pd.to_datetime(
        adm['discharge_time'], errors='coerce')

    # Drop nulls
    adm = adm.dropna(subset=['admit_time','discharge_time'])

    # Stratified sample by admission_type
    sampled = adm.groupby('admission_type', group_keys=False).apply(
        lambda x: x.sample(
            min(len(x),
                int(MAX_ENCOUNTERS * len(x) / len(adm)) + 1),
            random_state=42
        )
    ).head(MAX_ENCOUNTERS)

    enc_ids = set(sampled['encounter_id'].astype(int).tolist())
    pat_ids = set(sampled['patient_id'].astype(int).tolist())

    print(f"  Sampled {len(enc_ids):,} encounters "
          f"from {len(pat_ids):,} patients")
    print(f"  Admission types: "
          f"{sampled['admission_type'].value_counts().to_dict()}")

    return pat_ids, enc_ids, sampled

# ── 1. Patients ───────────────────────────────────────────────
def load_patients(pat_ids):
    print("\n[1/7] Loading patients...")
    df = pd.read_csv(find_file('hosp', 'patients'))
    df = df.rename(columns={'subject_id': 'patient_id'})
    df = df[df['patient_id'].isin(pat_ids)].copy()
    df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
    cols = ['patient_id','gender','anchor_age',
            'anchor_year','dod']
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'patient')

# ── 2. Encounters ─────────────────────────────────────────────
def load_encounters(sampled_df):
    print("\n[2/7] Loading encounters...")
    cols = [
        'encounter_id','patient_id','admit_time',
        'discharge_time','admission_type',
        'admission_location','discharge_location',
        'insurance','marital_status','race',
        'hospital_expire_flag'
    ]
    cols = [c for c in cols if c in sampled_df.columns]
    chunk_insert(sampled_df[cols], 'encounter')

# ── 3. ICU Stays ──────────────────────────────────────────────
def load_icu_stays(enc_ids):
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
    df = df[df['encounter_id'].isin(enc_ids)].copy()
    df['icu_intime']  = pd.to_datetime(
        df['icu_intime'],  errors='coerce')
    df['icu_outtime'] = pd.to_datetime(
        df['icu_outtime'], errors='coerce')
    cols = ['icu_stay_id','encounter_id','patient_id',
            'first_careunit','last_careunit',
            'icu_intime','icu_outtime','los_hours']
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'icu_stay')

# ── 4. Diagnoses ──────────────────────────────────────────────
def load_diagnoses(enc_ids):
    print("\n[4/7] Loading diagnoses...")
    df   = pd.read_csv(find_file('hosp', 'diagnoses_icd'))
    desc = pd.read_csv(find_file('hosp', 'd_icd_diagnoses'))
    desc = desc.rename(columns={'long_title': 'description'})
    df   = df[df['hadm_id'].isin(enc_ids)].copy()
    df   = df.merge(
        desc[['icd_code','icd_version','description']],
        on=['icd_code','icd_version'], how='left'
    )
    df   = df.rename(columns={'hadm_id': 'encounter_id'})
    df['description'] = df['description'].fillna('Unknown')
    cols = ['encounter_id','icd_code','icd_version',
            'description','seq_num']
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'diagnosis')

# ── 5. Medications ────────────────────────────────────────────
def load_medications(enc_ids):
    print("\n[5/7] Loading medications...")
    df = pd.read_csv(find_file('hosp', 'prescriptions'))
    df = df.rename(columns={
        'hadm_id':   'encounter_id',
        'drug':      'drug_name',
        'starttime': 'start_time',
        'stoptime':  'stop_time',
    })
    df = df.dropna(subset=['encounter_id'])
    df['encounter_id'] = df['encounter_id'].astype(int)
    df = df[df['encounter_id'].isin(enc_ids)].copy()
    df['start_time'] = pd.to_datetime(
        df['start_time'], errors='coerce')
    df['stop_time']  = pd.to_datetime(
        df['stop_time'],  errors='coerce')
    df['drug_name']  = df['drug_name'].fillna('Unknown')
    cols = ['encounter_id','drug_name','drug_type',
            'route','start_time','stop_time']
    cols = [c for c in cols if c in df.columns]
    chunk_insert(df[cols], 'medication')

# ── 6. Vitals ─────────────────────────────────────────────────
def load_vitals(enc_ids):
    print("\n[6/7] Loading vitals...")

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
        usecols=['hadm_id','stay_id','itemid',
                 'charttime','valuenum'],
        dtype={'hadm_id': 'Int64',
               'stay_id': 'Int64',
               'itemid':  'int32'},
        chunksize=500_000
    )

    total = 0
    for i, chunk in enumerate(chunks):
        # Filter to our encounters AND key vital items
        chunk = chunk[
            chunk['hadm_id'].isin(enc_ids) &
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
            out.to_sql('vital_sign', engine,
                       if_exists='append',
                       index=False,
                       method='multi',
                       chunksize=2000)
            total += len(out)

        print(f"  Chunk {i+1} | vitals: {total:,}", end='\r')

        # Stop early if we have enough vitals
        # ~50 readings per encounter × 10k encounters = 500k
        if total >= 500_000:
            print(f"\n  Reached 500k vitals cap — stopping early")
            break

    print(f"\n  Vitals loaded: {total:,} ✅")

# ── 7. Labs ───────────────────────────────────────────────────
def load_labs(enc_ids):
    print("\n[7/7] Loading labs...")

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
        50902: 'Chloride',
    }

    chunks = pd.read_csv(
        find_file('hosp', 'labevents'),
        usecols=['subject_id','hadm_id','itemid',
                 'charttime','value','valuenum',
                 'valueuom','flag'],
        dtype={'hadm_id':    'Int64',
               'subject_id': 'Int64',
               'itemid':     'int32'},
        chunksize=500_000
    )

    total = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk[
            chunk['hadm_id'].isin(enc_ids) &
            chunk['itemid'].isin(KEY_LABS)
        ].copy()
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
            'itemid','value','valuenum',
            'valueuom','flag','chart_time'
        ]]

        if not out.empty:
            out.to_sql('lab_result', engine,
                       if_exists='append',
                       index=False,
                       method='multi',
                       chunksize=2000)
            total += len(out)

        print(f"  Chunk {i+1} | labs: {total:,}", end='\r')

    print(f"\n  Labs loaded: {total:,} ✅")

# ── Create Indexes ────────────────────────────────────────────
def create_indexes():
    print("\n[Indexes] Creating indexes for faster queries...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_vital_enc ON vital_sign(encounter_id);",
        "CREATE INDEX IF NOT EXISTS idx_vital_type ON vital_sign(encounter_id, vital_type);",
        "CREATE INDEX IF NOT EXISTS idx_lab_enc ON lab_result(encounter_id);",
        "CREATE INDEX IF NOT EXISTS idx_lab_name ON lab_result(encounter_id, lab_name);",
        "CREATE INDEX IF NOT EXISTS idx_diag_enc ON diagnosis(encounter_id);",
        "CREATE INDEX IF NOT EXISTS idx_med_enc ON medication(encounter_id);",
        "CREATE INDEX IF NOT EXISTS idx_enc_pat ON encounter(patient_id);",
        "CREATE INDEX IF NOT EXISTS idx_enc_admit ON encounter(admit_time);",
        "CREATE INDEX IF NOT EXISTS idx_icu_enc ON icu_stay(encounter_id);",
    ]
    with engine.connect() as conn:
        for idx in indexes:
            conn.execute(text(idx))
            conn.commit()
    print("  Indexes created ✅")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("MediPredict ETL — MIMIC-IV Sampled Load")
    print(f"Target: {MAX_ENCOUNTERS:,} encounters")
    print("=" * 55)

    # Clear existing
    print("\nClearing existing data...")
    with engine.connect() as conn:
        conn.execute(text("""
            TRUNCATE TABLE
                risk_prediction, vital_sign,
                lab_result, medication,
                diagnosis, icu_stay,
                encounter, patient
            CASCADE;
        """))
        conn.commit()
    print("  Cleared ✅")

    # Sample
    pat_ids, enc_ids, sampled_df = sample_encounters()

    # Load each table
    load_patients(pat_ids)
    load_encounters(sampled_df)
    load_icu_stays(enc_ids)
    load_diagnoses(enc_ids)
    load_medications(enc_ids)
    load_vitals(enc_ids)
    load_labs(enc_ids)

    # Indexes
    create_indexes()

    # Final summary
    print("\n" + "=" * 55)
    print("✅ ETL Complete!")
    print("=" * 55)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT 'patients'    AS tbl, COUNT(*) AS n FROM patient
            UNION ALL SELECT 'encounters',  COUNT(*) FROM encounter
            UNION ALL SELECT 'icu_stays',   COUNT(*) FROM icu_stay
            UNION ALL SELECT 'diagnoses',   COUNT(*) FROM diagnosis
            UNION ALL SELECT 'medications', COUNT(*) FROM medication
            UNION ALL SELECT 'vitals',      COUNT(*) FROM vital_sign
            UNION ALL SELECT 'labs',        COUNT(*) FROM lab_result
        """))
        print(f"\n  {'Table':<15} {'Rows':>10}")
        print("  " + "-"*26)
        for r in rows:
            print(f"  {r[0]:<15} {r[1]:>10,}")

    # Check readmission rate
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(*)                              AS total,
                SUM(readmitted_30d)                   AS readmitted,
                ROUND(AVG(readmitted_30d)*100, 2)     AS rate_pct
            FROM readmission_label
        """)).fetchone()
        print(f"\n  Readmission label check:")
        print(f"  Total      : {row[0]:,}")
        print(f"  Readmitted : {row[1]:,}")
        print(f"  Rate       : {row[2]}%")