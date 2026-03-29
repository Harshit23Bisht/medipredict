-- ============================================================
-- MediPredict Schema v2 — MIMIC Compatible
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Patients
CREATE TABLE IF NOT EXISTS patient (
    patient_id      VARCHAR(20) PRIMARY KEY,  -- MIMIC subject_id
    gender          CHAR(1),
    anchor_age      INTEGER,                  -- MIMIC anonymized age
    anchor_year     INTEGER,
    dod             DATE                      -- date of death if applicable
);

-- Hospital Admissions (Encounters)
CREATE TABLE IF NOT EXISTS encounter (
    encounter_id        VARCHAR(20) PRIMARY KEY,  -- MIMIC hadm_id
    patient_id          VARCHAR(20) REFERENCES patient(patient_id),
    admit_time          TIMESTAMP,
    discharge_time      TIMESTAMP,
    admission_type      VARCHAR(50),
    admission_location  VARCHAR(100),
    discharge_location  VARCHAR(100),
    insurance           VARCHAR(50),
    marital_status      VARCHAR(50),
    race                VARCHAR(100),
    hospital_expire_flag SMALLINT              -- 1 if died in hospital
);

-- ICU Stays
CREATE TABLE IF NOT EXISTS icu_stay (
    icu_stay_id     VARCHAR(20) PRIMARY KEY,  -- MIMIC stay_id
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    patient_id      VARCHAR(20) REFERENCES patient(patient_id),
    icu_intime      TIMESTAMP,
    icu_outtime     TIMESTAMP,
    los_hours       FLOAT                     -- length of ICU stay
);

-- Diagnoses
CREATE TABLE IF NOT EXISTS diagnosis (
    id              SERIAL PRIMARY KEY,
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    icd_code        VARCHAR(20),
    icd_version     SMALLINT,                 -- 9 or 10
    description     TEXT,
    seq_num         INTEGER                   -- priority of diagnosis
);

-- Medications / Prescriptions
CREATE TABLE IF NOT EXISTS medication (
    id              SERIAL PRIMARY KEY,
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    drug_name       TEXT,
    drug_type       VARCHAR(50),
    route           VARCHAR(50),
    start_time      TIMESTAMP,
    stop_time       TIMESTAMP
);

-- Lab Results
CREATE TABLE IF NOT EXISTS lab_result (
    id              SERIAL PRIMARY KEY,
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    patient_id      VARCHAR(20) REFERENCES patient(patient_id),
    lab_name        VARCHAR(100),
    itemid          INTEGER,
    value           TEXT,
    valuenum        FLOAT,
    valueuom        VARCHAR(20),              -- unit of measure
    flag            VARCHAR(20),             -- abnormal/normal
    chart_time      TIMESTAMP
);

-- Vital Signs
CREATE TABLE IF NOT EXISTS vital_sign (
    id              SERIAL PRIMARY KEY,
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    icu_stay_id     VARCHAR(20),
    vital_type      VARCHAR(50),
    value           FLOAT,
    unit            VARCHAR(20),
    recorded_at     TIMESTAMP
);

-- Imaging Studies (metadata only — images in MongoDB)
CREATE TABLE IF NOT EXISTS imaging_study (
    study_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    patient_id      VARCHAR(20) REFERENCES patient(patient_id),
    modality        VARCHAR(50),
    body_part       VARCHAR(50),
    mongo_id        TEXT,                     -- MongoDB ObjectId
    file_name       TEXT,
    dicom_path      TEXT,
    study_date      DATE,
    report_text     TEXT                      -- radiologist report
);

-- Clinical Notes (metadata — full text in MongoDB)
CREATE TABLE IF NOT EXISTS clinical_note (
    note_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    patient_id      VARCHAR(20) REFERENCES patient(patient_id),
    note_type       VARCHAR(50),             -- discharge_summary, radiology etc
    mongo_id        TEXT,                    -- MongoDB ObjectId
    note_date       TIMESTAMP,
    is_error        BOOLEAN DEFAULT FALSE
);

-- Risk Predictions
CREATE TABLE IF NOT EXISTS risk_prediction (
    prediction_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    VARCHAR(20) REFERENCES encounter(encounter_id),
    xgb_score       FLOAT,
    lstm_score      FLOAT,
    cnn_score       FLOAT,
    fusion_score    FLOAT,                   -- final combined score
    risk_level      VARCHAR(10),
    model_version   VARCHAR(20),
    predicted_at    TIMESTAMP DEFAULT NOW()
);

-- ── Views ─────────────────────────────────────────────────────

-- Readmission Label
CREATE OR REPLACE VIEW readmission_label AS
SELECT
    e.encounter_id,
    e.patient_id,
    e.admit_time::date AS admit_date,
    e.discharge_time::date AS discharge_date,
    LEAD(e.admit_time) OVER (
        PARTITION BY e.patient_id ORDER BY e.admit_time
    )::date AS next_admit_date,
    CASE
        WHEN LEAD(e.admit_time) OVER (
            PARTITION BY e.patient_id ORDER BY e.admit_time
        ) - e.discharge_time <= INTERVAL '30 days'
        THEN 1 ELSE 0
    END AS readmitted_30d
FROM encounter e;

-- Encounter Features for XGBoost
CREATE OR REPLACE VIEW encounter_features AS
SELECT
    e.encounter_id,
    e.admit_time::date AS admit_date,
    p.anchor_age +
        (EXTRACT(YEAR FROM e.admit_time) - p.anchor_year) AS age_at_admission,
    CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,
    EXTRACT(EPOCH FROM (e.discharge_time - e.admit_time))/86400 AS length_of_stay,
    COALESCE((
        SELECT COUNT(*) FROM diagnosis d
        WHERE d.encounter_id = e.encounter_id
    ), 0) AS num_diagnoses,
    COALESCE((
        SELECT COUNT(DISTINCT drug_name) FROM medication m
        WHERE m.encounter_id = e.encounter_id
    ), 0) AS num_medications,
    COALESCE((
        SELECT AVG(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'heart_rate'
    ), 0) AS avg_hr,
    COALESCE((
        SELECT MAX(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'bp_systolic'
    ), 0) AS max_bp_sys,
    COALESCE((
        SELECT AVG(value) FROM vital_sign v
        WHERE v.encounter_id = e.encounter_id
        AND v.vital_type = 'temperature'
    ), 0) AS avg_temp,
    COALESCE((
        SELECT MAX(valuenum) FROM lab_result l
        WHERE l.encounter_id = e.encounter_id
        AND l.lab_name = 'Creatinine'
    ), 0) AS max_creatinine,
    COALESCE((
        SELECT MAX(valuenum) FROM lab_result l
        WHERE l.encounter_id = e.encounter_id
        AND l.lab_name = 'White Blood Cells'
    ), 0) AS max_wbc,
    COALESCE((
        SELECT COUNT(*) FROM encounter e2
        WHERE e2.patient_id = e.patient_id
        AND e2.admit_time < e.admit_time
    ), 0) AS num_prior_admissions
FROM encounter e
JOIN patient p ON p.patient_id = e.patient_id;