"""
ETL Step 1 — Load MIMIC-IV Demo CSVs into PostgreSQL
Run: python etl/01_load_postgres.py
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))
RAW = os.getenv("MIMIC_RAW_PATH")


# 🔥 helper for fixing ID columns
def clean_id(col):
    return col.astype('Int64').astype(str)


def load_patients():
    print("Loading patients...")
    df = pd.read_csv(f"{RAW}/hosp/patients.csv")

    df = df.rename(columns={'subject_id': 'patient_id'})
    df['dod'] = pd.to_datetime(df['dod'], errors='coerce')

    df[['patient_id','gender','anchor_age',
        'anchor_year','dod']].to_sql(
        'patient', engine,
        if_exists='append', index=False, method='multi')

    print(f"  Loaded {len(df)} patients")


def load_encounters():
    print("Loading encounters...")
    df = pd.read_csv(f"{RAW}/hosp/admissions.csv")

    df = df.rename(columns={
        'subject_id':'patient_id',
        'hadm_id':'encounter_id',
        'admittime':'admit_time',
        'dischtime':'discharge_time',
    })

    df['admit_time'] = pd.to_datetime(df['admit_time'])
    df['discharge_time'] = pd.to_datetime(df['discharge_time'])

    # 🔥 FIX IDs
    df['encounter_id'] = clean_id(df['encounter_id'])
    df['patient_id'] = clean_id(df['patient_id'])

    df = df[df['encounter_id'] != '<NA>']
    df = df[df['patient_id'] != '<NA>']

    cols = ['encounter_id','patient_id','admit_time',
            'discharge_time','admission_type',
            'admission_location','discharge_location',
            'insurance','marital_status','race',
            'hospital_expire_flag']

    df[cols].to_sql('encounter', engine,
                    if_exists='append', index=False, method='multi')

    print(f"  Loaded {len(df)} encounters")


def load_diagnoses():
    print("Loading diagnoses...")
    df = pd.read_csv(f"{RAW}/hosp/diagnoses_icd.csv")
    desc = pd.read_csv(f"{RAW}/hosp/d_icd_diagnoses.csv")

    df = df.merge(desc[['icd_code','long_title']],
                  on='icd_code', how='left')

    df = df.rename(columns={
        'hadm_id':'encounter_id',
        'long_title':'description'
    })

    df['encounter_id'] = clean_id(df['encounter_id'])
    df = df[df['encounter_id'] != '<NA>']

    df[['encounter_id','icd_code','icd_version',
        'description','seq_num']].to_sql(
        'diagnosis', engine,
        if_exists='append',
        index=False, method='multi', chunksize=1000)

    print(f"  Loaded {len(df)} diagnoses")


def load_medications():
    print("Loading medications...")
    df = pd.read_csv(f"{RAW}/hosp/prescriptions.csv")

    df = df.rename(columns={
        'hadm_id':'encounter_id',
        'drug':'drug_name',
        'starttime':'start_time',
        'stoptime':'stop_time',
    })

    df['encounter_id'] = clean_id(df['encounter_id'])

    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['stop_time'] = pd.to_datetime(df['stop_time'], errors='coerce')

    df = df[df['encounter_id'] != '<NA>']

    df[['encounter_id','drug_name','drug_type',
        'route','start_time','stop_time']].to_sql(
        'medication', engine,
        if_exists='append',
        index=False, method='multi', chunksize=1000)

    print(f"  Loaded {len(df)} medications")


# ✅ FIXED VITALS
def load_vitals():
    print("Loading vitals (this may take a while)...")

    VITAL_ITEMS = {
        220045: 'heart_rate',
        220179: 'bp_systolic',
        220180: 'bp_diastolic',
        223761: 'temperature',
        220210: 'respiratory_rate',
        220277: 'spo2',
    }

    df = pd.read_csv(
        f"{RAW}/icu/chartevents.csv",
        usecols=['hadm_id','stay_id','itemid',
                 'charttime','value','valuenum']
    )

    df = df[df['itemid'].isin(VITAL_ITEMS.keys())]

    df['vital_type'] = df['itemid'].map(VITAL_ITEMS)

    df['encounter_id'] = clean_id(df['hadm_id'])
    df['icu_stay_id'] = clean_id(df['stay_id'])

    df = df.rename(columns={'charttime':'recorded_at'})

    df['value'] = pd.to_numeric(df['valuenum'], errors='coerce')
    df['recorded_at'] = pd.to_datetime(df['recorded_at'])

    df = df.dropna(subset=['value','encounter_id'])
    df = df[df['encounter_id'] != '<NA>']

    df[['encounter_id','icu_stay_id','vital_type',
        'value','recorded_at']].to_sql(
        'vital_sign', engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=5000
    )

    print(f"  Loaded {len(df)} vital readings")


# ✅ FIXED LABS
def load_labs():
    print("Loading lab results...")

    df = pd.read_csv(
        f"{RAW}/hosp/labevents.csv",
        usecols=['hadm_id','subject_id','itemid',
                 'charttime','value','valuenum',
                 'valueuom','flag']
    )

    df = df.rename(columns={
        'hadm_id':'encounter_id',
        'subject_id':'patient_id',
        'charttime':'chart_time',
    })

    df['encounter_id'] = clean_id(df['encounter_id'])
    df['patient_id'] = clean_id(df['patient_id'])

    df['chart_time'] = pd.to_datetime(df['chart_time'], errors='coerce')
    df['lab_name'] = 'lab_' + df['itemid'].astype(str)

    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')

    df = df.dropna(subset=['valuenum','encounter_id','patient_id'])

    df = df[df['encounter_id'] != '<NA>']
    df = df[df['patient_id'] != '<NA>']

    df[['encounter_id','patient_id','lab_name',
        'itemid','valuenum','valueuom',
        'flag','chart_time']].to_sql(
        'lab_result', engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=5000
    )

    print(f"  Loaded {len(df)} lab results")


if __name__ == "__main__":
    load_patients()
    load_encounters()
    load_diagnoses()
    load_medications()
    load_vitals()
    load_labs()

    print("\n✅ All data loaded into PostgreSQL!")