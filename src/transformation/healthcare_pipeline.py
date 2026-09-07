import os
import pandas as pd

RAW_PATH = "data/raw/patient_encounters.csv"
BRONZE_PATH = "data/processed/bronze_patient_encounters.csv"
SILVER_PATH = "data/processed/silver_patient_encounters.csv"
GOLD_PATH = "data/processed/gold_healthcare_analytics.csv"

os.makedirs("data/processed", exist_ok=True)

print("\n=== INTELLIGENT HEALTHCARE DATA PLATFORM ===")

# ---------------- BRONZE LAYER ----------------
print("\n[BRONZE] Ingesting raw healthcare data...")

bronze = pd.read_csv(RAW_PATH)
bronze["ingestion_timestamp"] = pd.Timestamp.now()

bronze.to_csv(BRONZE_PATH, index=False)

print(f"Bronze records: {len(bronze)}")


# ---------------- SILVER LAYER ----------------
print("\n[SILVER] Cleaning and transforming healthcare data...")

silver = bronze.copy()

silver = silver.drop_duplicates(subset=["patient_id"])
silver = silver.dropna()

silver["admission_date"] = pd.to_datetime(silver["admission_date"])

silver["age_group"] = pd.cut(
    silver["age"],
    bins=[0, 39, 59, 74, 120],
    labels=["18-39", "40-59", "60-74", "75+"]
)

silver["cost_per_day"] = (
    silver["total_cost"] / silver["length_of_stay"]
).round(2)

silver["clinical_risk_score"] = (
    silver["previous_admissions"] * 2
    + silver["abnormal_lab_results"] * 1.5
    + silver["num_medications"] * 0.3
    + (silver["age"] >= 65).astype(int) * 2
).round(2)

silver["risk_category"] = pd.cut(
    silver["clinical_risk_score"],
    bins=[-1, 7, 14, float("inf")],
    labels=["Low", "Medium", "High"]
)

silver.to_csv(SILVER_PATH, index=False)

print(f"Silver records: {len(silver)}")


# ---------------- GOLD LAYER ----------------
print("\n[GOLD] Building healthcare analytics dataset...")

gold = (
    silver.groupby(["diagnosis", "risk_category"], observed=True)
    .agg(
        patient_count=("patient_id", "count"),
        avg_age=("age", "mean"),
        avg_length_of_stay=("length_of_stay", "mean"),
        avg_total_cost=("total_cost", "mean"),
        avg_previous_admissions=("previous_admissions", "mean"),
        readmission_rate=("readmitted_30_days", "mean")
    )
    .reset_index()
)

gold["avg_age"] = gold["avg_age"].round(1)
gold["avg_length_of_stay"] = gold["avg_length_of_stay"].round(2)
gold["avg_total_cost"] = gold["avg_total_cost"].round(2)
gold["avg_previous_admissions"] = gold["avg_previous_admissions"].round(2)
gold["readmission_rate"] = (gold["readmission_rate"] * 100).round(2)

gold.to_csv(GOLD_PATH, index=False)

print(f"Gold analytics rows: {len(gold)}")

print("\nPipeline completed successfully!")
print("\nSample Gold Analytics:")
print(gold.head(10).to_string(index=False))
