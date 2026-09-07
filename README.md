# Intelligent Healthcare Data & AI Platform

An end-to-end healthcare data, machine learning, and Generative AI platform for analyzing patient encounters, predicting 30-day hospital readmission risk, and generating grounded healthcare analytics insights.

> This project uses fully synthetic healthcare data and is intended for engineering and portfolio demonstration purposes only. It is not a clinical decision-support system.

## Business Problem

Hospital readmissions can increase healthcare costs and may indicate gaps in discharge planning and follow-up care. This project demonstrates how healthcare encounter data can be transformed into analytics-ready datasets, used to train a machine learning risk model, and connected to a Retrieval-Augmented Generation (RAG) assistant for grounded analytical Q&A.

## Architecture
```text
Synthetic Patient Data
        |
        v
Bronze Raw Layer
        |
        v
Silver Curated Layer
        |
        v
Gold Analytics Layer
       / \
      /   \
     v     v
ML Model   RAG / GenAI
   |          |
   v          v
FastAPI    Nomic Embeddings
              |
              v
       Semantic Retrieval
              |
              v
       Llama 3.2 / Ollama

Databricks/PySpark + Delta Lake -> scalable data pipeline implementation
Snowflake -> analytics warehouse implementation
```
## Technology Stack

Data Engineering: Python, Pandas, PySpark, Databricks, Delta Lake, Medallion Architecture

Machine Learning: scikit-learn, Random Forest, feature engineering, preprocessing, model evaluation, Joblib

Generative AI / RAG: Llama 3.2, Ollama, Nomic Embed Text, embeddings, cosine similarity, semantic retrieval, grounded generation

API: FastAPI, Pydantic, Uvicorn, REST

Data Warehouse: Snowflake SQL

Cloud-ready architecture: AWS, Azure, and GCP object storage and managed data/AI services

## Data Engineering Pipeline

The local pipeline follows a Bronze, Silver, and Gold medallion architecture.

Bronze Layer:
- Ingests synthetic patient encounter data
- Adds ingestion metadata
- Preserves source-level records

Silver Layer:
- Removes duplicate and invalid records
- Performs data validation and type conversion
- Creates age groups and cost-per-day features
- Calculates clinical risk scores and risk categories

Gold Layer:
- Produces diagnosis and risk-level analytics
- Calculates patient counts, average costs, length of stay, previous admissions, and readmission rates

A separate PySpark and Delta Lake implementation is included for Databricks.

## Machine Learning

A Random Forest classifier predicts 30-day hospital readmission risk using patient encounter and clinical features.

Features include:
- Age
- Gender
- Diagnosis
- Previous admissions
- Length of stay
- Number of medications
- Abnormal lab results
- Insurance type
- Total cost

## Model Performance

Accuracy: 0.72
Precision: 0.7615
Recall: 0.7984
F1 Score: 0.7795
ROC-AUC: 0.7696

The trained model is persisted with Joblib and served through a FastAPI REST service.

## FastAPI Prediction Service

The /predict endpoint accepts patient encounter information and returns:

- Readmission prediction
- Readmission probability
- Risk level
- Follow-up recommendation

Example tested result:

Prediction: Readmission
Probability: 86.31%
Risk Level: HIGH
Recommendation: Prioritize discharge follow-up and care coordination.

## Generative AI and RAG

The project implements a fully local Retrieval-Augmented Generation pipeline using Ollama.

RAG flow:

User Question
     |
     v
Nomic Embed Text
     |
     v
Vector Embedding
     |
     v
Cosine Similarity Retrieval
     |
     v
Relevant Gold Analytics Context
     |
     v
Llama 3.2
     |
     v
Grounded Healthcare Answer

The RAG assistant retrieves relevant healthcare analytics records before sending context to the LLM, reducing unsupported responses and grounding answers in the processed dataset.

Tested question:

Which patient group has the highest readmission rate?

Grounded answer:

Heart Disease patients in the High risk category, with a readmission rate of 85.15%.
## Databricks and PySpark

The project includes a Databricks-compatible PySpark implementation in:

notebooks/databricks_healthcare_pipeline.py

It demonstrates:
- Explicit Spark schema
- Bronze ingestion
- Silver transformations
- Clinical risk feature engineering
- Gold-level aggregations
- Delta table writes

The Databricks pipeline is implementation-ready code. It was not executed against a live Databricks workspace as part of this local demo.

## Snowflake

The Snowflake implementation is available in:

snowflake/healthcare_warehouse.sql

It includes:
- Healthcare database and analytics schema
- Patient encounter table
- Readmission analytics table
- High-risk patient view
- Healthcare KPI view
- Analytical SQL queries

The SQL was designed for Snowflake but was not deployed to a live Snowflake account as part of this local demo.

## Project Structure

healthcare-ai-data-platform/
  data/
    raw/
    processed/
  notebooks/
    databricks_healthcare_pipeline.py
  snowflake/
    healthcare_warehouse.sql
  src/
    ingestion/
    transformation/
    ml/
    genai/
    api/
  tests/
  architecture/
  requirements.txt
  .gitignore
  README.md

## Running the Project

1. Create and activate the Python environment.

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Generate synthetic healthcare data.

python src/ingestion/generate_healthcare_data.py

3. Run the Bronze/Silver/Gold pipeline.

python src/transformation/healthcare_pipeline.py

4. Train the readmission model.

python src/ml/train_readmission_model.py

5. Start FastAPI.

uvicorn src.api.main:app --reload

Swagger UI:

http://127.0.0.1:8000/docs

6. Run the RAG assistant.

python src/genai/rag_healthcare_assistant.py

## Local LLM Setup

This project uses Ollama locally and does not require a paid LLM API key.

Required models:

llama3.2:3b
nomic-embed-text

Install them with:

ollama pull llama3.2:3b
ollama pull nomic-embed-text

## Production Extensions

The architecture can be extended with:
- AWS S3, Azure Data Lake Storage, or Google Cloud Storage
- Managed Databricks and Unity Catalog
- Snowflake cloud data warehouse
- MLflow model tracking and registry
- Docker and Kubernetes
- CI/CD pipelines
- Kafka streaming
- IAM/RBAC and secrets management
- Encryption and PHI/PII controls
- Model monitoring and drift detection
- Centralized logging and observability

## Disclaimer

All patient records used in this project are synthetically generated. No real patient data or protected health information (PHI) is used.

The machine learning predictions and Generative AI responses are provided strictly for software engineering and portfolio demonstration purposes and must not be used for medical diagnosis or clinical decision-making.
