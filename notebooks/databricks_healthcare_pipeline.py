# Databricks notebook source
# Intelligent Healthcare Data & AI Platform
# PySpark Medallion Architecture: Bronze -> Silver -> Gold

from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# BRONZE LAYER
# In production this path can point to ADLS, S3 or GCS.

raw_path = "/Volumes/healthcare/raw/patient_encounters.csv"

schema = StructType([
    StructField("patient_id", StringType(), False),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("diagnosis", StringType(), True),
    StructField("previous_admissions", IntegerType(), True),
    StructField("length_of_stay", IntegerType(), True),
    StructField("num_medications", IntegerType(), True),
    StructField("abnormal_lab_results", IntegerType(), True),
    StructField("insurance_type", StringType(), True),
    StructField("total_cost", DoubleType(), True),
    StructField("admission_date", DateType(), True),
    StructField("readmitted_30_days", IntegerType(), True)
])

bronze_df = (
    spark.read
    .option("header", True)
    .schema(schema)
    .csv(raw_path)
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("source_system", F.lit("hospital_ehr"))
)

bronze_df.write.format("delta").mode("overwrite").saveAsTable(
    "healthcare.bronze.patient_encounters"
)

print("Bronze layer completed")

# COMMAND ----------

# SILVER LAYER

silver_df = (
    bronze_df
    .dropDuplicates(["patient_id"])
    .dropna(subset=["patient_id", "age", "diagnosis"])
    .filter((F.col("age") >= 18) & (F.col("age") <= 120))
    .withColumn(
        "age_group",
        F.when(F.col("age") < 40, "18-39")
         .when(F.col("age") < 60, "40-59")
         .when(F.col("age") < 75, "60-74")
         .otherwise("75+")
    )
    .withColumn(
        "cost_per_day",
        F.round(F.col("total_cost") / F.col("length_of_stay"), 2)
    )
    .withColumn(
        "clinical_risk_score",
        F.round(
            F.col("previous_admissions") * 2
            + F.col("abnormal_lab_results") * 1.5
            + F.col("num_medications") * 0.3
            + F.when(F.col("age") >= 65, 2).otherwise(0),
            2
        )
    )
)

silver_df = silver_df.withColumn(
    "risk_category",
    F.when(F.col("clinical_risk_score") <= 7, "Low")
     .when(F.col("clinical_risk_score") <= 14, "Medium")
     .otherwise("High")
)

silver_df.write.format("delta").mode("overwrite").saveAsTable(
    "healthcare.silver.patient_encounters"
)

print("Silver layer completed")

# COMMAND ----------

# GOLD LAYER

gold_df = (
    silver_df
    .groupBy("diagnosis", "risk_category")
    .agg(
        F.count("patient_id").alias("patient_count"),
        F.round(F.avg("age"), 1).alias("avg_age"),
        F.round(F.avg("length_of_stay"), 2).alias("avg_length_of_stay"),
        F.round(F.avg("total_cost"), 2).alias("avg_total_cost"),
        F.round(F.avg("previous_admissions"), 2).alias(
            "avg_previous_admissions"
        ),
        F.round(
            F.avg("readmitted_30_days") * 100,
            2
        ).alias("readmission_rate")
    )
)

gold_df.write.format("delta").mode("overwrite").saveAsTable(
    "healthcare.gold.readmission_analytics"
)

print("Gold layer completed")

display(
    gold_df.orderBy(F.desc("readmission_rate"))
)
