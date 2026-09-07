# Intelligent Healthcare Data & AI Platform — Architecture

```mermaid
flowchart TD
    A[Synthetic Patient Encounter Data] --> B[Bronze Layer<br/>Raw Ingestion]

    B --> C[Silver Layer<br/>Validation & Transformation]
    C --> D[Gold Layer<br/>Healthcare Analytics]

    D --> E[Machine Learning]
    D --> F[Generative AI / RAG]
    D --> G[Snowflake Analytics]

    E --> H[Random Forest<br/>Readmission Model]
    H --> I[FastAPI Prediction Service]

    F --> J[Nomic Embeddings]
    J --> K[Vector Similarity Retrieval]
    K --> L[Llama 3.2 via Ollama]
    L --> M[Grounded Healthcare Insights]

    B -. scalable implementation .-> N[Databricks / PySpark]
    C -. Delta Lake .-> N
    D -. Gold Tables .-> N

    O[Cloud Object Storage<br/>AWS S3 / Azure ADLS / GCP GCS] -. Production Extension .-> B
```

## Data Flow

**Synthetic Patient Data → Bronze → Silver → Gold → ML / RAG / Analytics**

- **Bronze:** Raw healthcare encounter ingestion
- **Silver:** Data validation, cleaning and feature engineering
- **Gold:** Aggregated healthcare analytics and readmission KPIs
- **ML:** Random Forest model for 30-day readmission prediction
- **API:** FastAPI exposes real-time prediction endpoints
- **GenAI/RAG:** Embeddings + semantic retrieval + local Llama 3.2
- **Databricks:** PySpark/Delta Lake implementation for scalable processing
- **Snowflake:** Analytical warehouse SQL implementation
- **Cloud:** AWS, Azure and GCP-ready storage architecture

> All patient data used by this project is synthetically generated.
