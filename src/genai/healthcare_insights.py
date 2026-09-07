import pandas as pd

GOLD_PATH = "data/processed/gold_healthcare_analytics.csv"

df = pd.read_csv(GOLD_PATH)


def healthcare_insight(question: str) -> str:
    q = question.lower()

    if "highest" in q and "readmission" in q:
        row = df.loc[df["readmission_rate"].idxmax()]

        return (
            f"The highest readmission rate is for {row['diagnosis']} "
            f"patients in the {row['risk_category']} risk category. "
            f"The readmission rate is {row['readmission_rate']:.2f}%, "
            f"with an average length of stay of "
            f"{row['avg_length_of_stay']:.2f} days and average cost of "
            f"${row['avg_total_cost']:,.2f}."
        )

    if "cost" in q:
        row = df.loc[df["avg_total_cost"].idxmax()]

        return (
            f"The highest average treatment cost is associated with "
            f"{row['diagnosis']} patients in the "
            f"{row['risk_category']} risk category, averaging "
            f"${row['avg_total_cost']:,.2f} per encounter. "
            f"Their readmission rate is {row['readmission_rate']:.2f}%."
        )

    if "high risk" in q or "high-risk" in q:
        high_risk = df[df["risk_category"] == "High"].copy()
        high_risk = high_risk.sort_values(
            "readmission_rate",
            ascending=False
        )

        top = high_risk.iloc[0]

        return (
            f"Among high-risk patients, {top['diagnosis']} currently "
            f"shows the highest readmission rate at "
            f"{top['readmission_rate']:.2f}%. "
            f"This group has {int(top['patient_count'])} patients "
            f"in the analytics dataset."
        )

    return (
        "I can analyze readmission risk, high-risk patient groups, "
        "treatment costs, length of stay, and diagnosis-level trends "
        "using the curated healthcare analytics data."
    )


if __name__ == "__main__":
    print("\n=== HEALTHCARE GENAI ANALYTICS ASSISTANT ===\n")

    questions = [
        "Which patient group has the highest readmission rate?",
        "Which group has the highest healthcare cost?",
        "Tell me about high risk patients."
    ]

    for question in questions:
        print(f"Question: {question}")
        print(f"AI Insight: {healthcare_insight(question)}")
        print()
