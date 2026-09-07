import csv
import random
from datetime import datetime, timedelta

random.seed(42)

diagnoses = ["Diabetes", "Hypertension", "Heart Disease", "COPD", "Kidney Disease"]
genders = ["Male", "Female"]
insurance_types = ["Medicare", "Medicaid", "Private", "Self-Pay"]

output_file = "data/raw/patient_encounters.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "patient_id",
        "age",
        "gender",
        "diagnosis",
        "previous_admissions",
        "length_of_stay",
        "num_medications",
        "abnormal_lab_results",
        "insurance_type",
        "total_cost",
        "admission_date",
        "readmitted_30_days"
    ])

    for patient_id in range(1, 1001):
        age = random.randint(18, 90)
        gender = random.choice(genders)
        diagnosis = random.choice(diagnoses)
        previous_admissions = random.randint(0, 6)
        length_of_stay = random.randint(1, 15)
        num_medications = random.randint(1, 15)
        abnormal_labs = random.randint(0, 8)
        insurance = random.choice(insurance_types)

        total_cost = round(
            1500
            + length_of_stay * random.uniform(500, 1500)
            + num_medications * random.uniform(20, 100),
            2
        )

        admission_date = (
            datetime(2025, 1, 1)
            + timedelta(days=random.randint(0, 364))
        ).strftime("%Y-%m-%d")

        risk_score = (
            previous_admissions * 0.12
            + abnormal_labs * 0.05
            + (0.15 if age >= 65 else 0)
            + (0.10 if length_of_stay >= 8 else 0)
        )

        readmitted = 1 if random.random() < min(risk_score, 0.90) else 0

        writer.writerow([
            f"P{patient_id:05d}",
            age,
            gender,
            diagnosis,
            previous_admissions,
            length_of_stay,
            num_medications,
            abnormal_labs,
            insurance,
            total_cost,
            admission_date,
            readmitted
        ])

print(f"Created 1,000 synthetic healthcare records: {output_file}")
