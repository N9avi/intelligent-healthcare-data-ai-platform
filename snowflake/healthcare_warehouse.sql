-- ============================================================
-- INTELLIGENT HEALTHCARE DATA & AI PLATFORM
-- Snowflake Analytics Warehouse
-- ============================================================

CREATE DATABASE IF NOT EXISTS HEALTHCARE_AI_DB;

USE DATABASE HEALTHCARE_AI_DB;

CREATE SCHEMA IF NOT EXISTS ANALYTICS;

USE SCHEMA ANALYTICS;

-- ------------------------------------------------------------
-- Healthcare Patient Encounter Table
-- ------------------------------------------------------------

CREATE OR REPLACE TABLE PATIENT_ENCOUNTERS (
    PATIENT_ID VARCHAR(20),
    AGE INTEGER,
    GENDER VARCHAR(20),
    DIAGNOSIS VARCHAR(100),
    PREVIOUS_ADMISSIONS INTEGER,
    LENGTH_OF_STAY INTEGER,
    NUM_MEDICATIONS INTEGER,
    ABNORMAL_LAB_RESULTS INTEGER,
    INSURANCE_TYPE VARCHAR(50),
    TOTAL_COST NUMBER(12,2),
    ADMISSION_DATE DATE,
    READMITTED_30_DAYS INTEGER,
    AGE_GROUP VARCHAR(20),
    COST_PER_DAY NUMBER(12,2),
    CLINICAL_RISK_SCORE NUMBER(10,2),
    RISK_CATEGORY VARCHAR(20),
    LOAD_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ------------------------------------------------------------
-- Gold Analytics Table
-- ------------------------------------------------------------

CREATE OR REPLACE TABLE READMISSION_ANALYTICS (
    DIAGNOSIS VARCHAR(100),
    RISK_CATEGORY VARCHAR(20),
    PATIENT_COUNT INTEGER,
    AVG_AGE NUMBER(6,2),
    AVG_LENGTH_OF_STAY NUMBER(8,2),
    AVG_TOTAL_COST NUMBER(12,2),
    AVG_PREVIOUS_ADMISSIONS NUMBER(8,2),
    READMISSION_RATE NUMBER(8,2),
    REFRESH_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ------------------------------------------------------------
-- High-Risk Patient Analytics View
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW HIGH_RISK_PATIENTS AS
SELECT
    PATIENT_ID,
    AGE,
    GENDER,
    DIAGNOSIS,
    PREVIOUS_ADMISSIONS,
    LENGTH_OF_STAY,
    ABNORMAL_LAB_RESULTS,
    TOTAL_COST,
    CLINICAL_RISK_SCORE,
    RISK_CATEGORY,
    READMITTED_30_DAYS
FROM PATIENT_ENCOUNTERS
WHERE RISK_CATEGORY = 'High';

-- ------------------------------------------------------------
-- Executive Healthcare KPI View
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW HEALTHCARE_KPI_SUMMARY AS
SELECT
    DIAGNOSIS,
    COUNT(*) AS TOTAL_PATIENTS,
    ROUND(AVG(TOTAL_COST), 2) AS AVG_TREATMENT_COST,
    ROUND(AVG(LENGTH_OF_STAY), 2) AS AVG_LENGTH_OF_STAY,
    ROUND(AVG(READMITTED_30_DAYS) * 100, 2) AS READMISSION_RATE,
    COUNT_IF(RISK_CATEGORY = 'High') AS HIGH_RISK_PATIENTS
FROM PATIENT_ENCOUNTERS
GROUP BY DIAGNOSIS;

-- ------------------------------------------------------------
-- Example Business Analytics Query
-- ------------------------------------------------------------

SELECT
    DIAGNOSIS,
    TOTAL_PATIENTS,
    HIGH_RISK_PATIENTS,
    AVG_TREATMENT_COST,
    READMISSION_RATE
FROM HEALTHCARE_KPI_SUMMARY
ORDER BY READMISSION_RATE DESC;
