-- ============================================================
-- MediPredict Schema v3 — Correct MIMIC-IV Types
-- ============================================================

-- Patients
CREATE TABLE IF NOT EXISTS patient (
    patient_id      INTEGER PRIMARY KEY,
    gender          CHAR(1),
    anchor_age      INTEGER,
    anchor_year     INTEGER,
    dod             DATE
);

-- Encounters (Hospital Admissions)
CREATE TABLE IF NOT EXISTS encounter (
    encounter_id        INTEGER PRIMARY KEY,
    patient_id          INTEGER REFERENCES patient(patient_id),
    admit_time          TIMESTAMP,
    discharge_time      TIMESTAMP,
    admission_type      VARCHAR(100),
    admission_location  VARCHAR(100),
    discharge_location  VARCHAR(100),
    insurance           VARCHAR(50),
    marital_status      VARCHAR(50),
    race                VARCHAR(100),
    hospital_expire_flag SMALLINT
);

-- ICU Stays
CREATE TABLE IF NOT EXISTS icu_stay (
    icu_stay_id     INTEGER PRIMARY KEY,
    encounter_id    INTEGER REFERENCES encounter(encounter_id),
    patient_id      INTEGER REFERENCES patient(patient_id),
    first_careunit  VARCHAR(50),
    last_careunit   VARCHAR(50),
    icu_intime      TIMESTAMP,
    icu_outtime     TIMESTAMP,
    los_hours       FLOAT
);

-- Diagnoses
CREATE TABLE IF NOT EXISTS diagnosis (
    id              SERIAL PRIMARY KEY,
    encounter_id    INTEGER REFERENCES encounter(encounter_id),
    icd_code        VARCHAR(20),
    icd_version     SMALLINT,
    description     TEXT,
    seq_num         INTEGER
);

-- Medications
CREATE TABLE IF NOT EXISTS medication (
    id              SERIAL PRIMARY KEY,
    encounter_id    INTEGER REFERENCES encounter(encounter_id),
    drug_name       TEXT,
    drug_type       VARCHAR(50),
    route           VARCHAR(50),
    start_time      TIMESTAMP,
    stop_time       TIMESTAMP
);

-- Lab Results
CREATE TABLE IF NOT EXISTS lab_result (
    id              SERIAL PRIMARY KEY,
    encounter_id    INTEGER,
    patient_id      INTEGER,
    lab_name        VARCHAR(100),
    itemid          INTEGER,
    value           TEXT,
    valuenum        FLOAT,
    valueuom        VARCHAR(20),
    flag            VARCHAR(20),
    chart_time      TIMESTAMP
);

-- Vital Signs
CREATE TABLE IF NOT EXISTS vital_sign (
    id              SERIAL PRIMARY KEY,
    encounter_id    INTEGER,
    icu_stay_id     INTEGER,
    vital_type      VARCHAR(50),
    value           FLOAT,
    unit            VARCHAR(20),
    recorded_at     TIMESTAMP
);

-- Imaging Studies (MongoDB reference)
CREATE TABLE IF NOT EXISTS imaging_study (
    study_id        SERIAL PRIMARY KEY,
    encounter_id    INTEGER,
    patient_id      INTEGER,
    modality        VARCHAR(50),
    body_part       VARCHAR(50),
    mongo_id        TEXT,
    file_name       TEXT,
    study_date      DATE,
    report_text     TEXT
);

-- Clinical Notes (MongoDB reference)
CREATE TABLE IF NOT EXISTS clinical_note (
    note_id         SERIAL PRIMARY KEY,
    encounter_id    INTEGER,
    patient_id      INTEGER,
    note_type       VARCHAR(50),
    mongo_id        TEXT,
    note_date       TIMESTAMP
);

-- Risk Predictions
CREATE TABLE IF NOT EXISTS risk_prediction (
    prediction_id   SERIAL PRIMARY KEY,
    encounter_id    INTEGER,
    xgb_score       FLOAT,
    lstm_score      FLOAT,
    cnn_score       FLOAT,
    fusion_score    FLOAT,
    risk_level      VARCHAR(10),
    model_version   VARCHAR(20),
    predicted_at    TIMESTAMP DEFAULT NOW()
);

-- ── Views ──────────────────────────────────────────────────

-- Readmission Label using LEAD window function
CREATE OR REPLACE VIEW readmission_label AS
SELECT
    e.encounter_id,
    e.patient_id,
    e.admit_time::date        AS admit_date,
    e.discharge_time::date    AS discharge_date,
    LEAD(e.admit_time) OVER (
        PARTITION BY e.patient_id
        ORDER BY e.admit_time
    )::date                   AS next_admit_date,
    CASE
        WHEN LEAD(e.admit_time) OVER (
            PARTITION BY e.patient_id
            ORDER BY e.admit_time
        ) - e.discharge_time <= INTERVAL '30 days'
        THEN 1 ELSE 0
    END                       AS readmitted_30d
FROM encounter e
WHERE e.discharge_time IS NOT NULL;

-- Encounter Features for XGBoost
CREATE OR REPLACE VIEW encounter_features AS
SELECT
    e.encounter_id,
    e.admit_time::date AS admit_date,

    -- Demographics
    COALESCE(
        p.anchor_age + (
            EXTRACT(YEAR FROM e.admit_time) - p.anchor_year
        ), 0
    )::INTEGER                AS age_at_admission,
    CASE WHEN p.gender = 'M'
         THEN 1 ELSE 0
    END                       AS gender,

    -- Encounter
    COALESCE(
        EXTRACT(EPOCH FROM (
            e.discharge_time - e.admit_time
        )) / 86400.0, 0
    )::FLOAT                  AS length_of_stay,

    -- Diagnoses
    COALESCE((
        SELECT COUNT(*) FROM diagnosis d
        WHERE d.encounter_id = e.encounter_id
    ), 0)                     AS num_diagnoses,

    -- Medications
    COALESCE((
        SELECT COUNT(DISTINCT drug_name) FROM medication m
        WHERE m.encounter_id = e.encounter_id
    ), 0)                     AS num_medications,

    -- Vitals
    COALESCE((
        SELECT AVG(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'heart_rate'
    ), 0)                     AS avg_hr,

    COALESCE((
        SELECT MAX(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'bp_systolic'
    ), 0)                     AS max_bp_sys,

    COALESCE((
        SELECT AVG(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'temperature_f'
    ), 0)                     AS avg_temp,

    -- Lab Results
    COALESCE((
        SELECT MAX(valuenum) FROM lab_result l
        WHERE l.encounter_id = e.encounter_id
        AND l.lab_name = 'Creatinine'
    ), 0)                     AS max_creatinine,

    COALESCE((
        SELECT MAX(valuenum) FROM lab_result l
        WHERE l.encounter_id = e.encounter_id
        AND l.lab_name = 'White Blood Cells'
    ), 0)                     AS max_wbc,

    -- History
    COALESCE((
        SELECT COUNT(*) FROM encounter e2
        WHERE e2.patient_id = e.patient_id
        AND e2.admit_time < e.admit_time
    ), 0)                     AS num_prior_admissions

FROM encounter e
JOIN patient p ON p.patient_id = e.patient_id
WHERE e.admit_time IS NOT NULL
AND   e.discharge_time IS NOT NULL;