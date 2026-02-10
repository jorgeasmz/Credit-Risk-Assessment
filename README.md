# Credit Risk Assessment System

A machine learning microservice and UI for evaluating credit applicant risk using the German Credit Data set.

**Live Demo:** [Streamlit App](https://jorgeasmz-credit-risk-assessment.streamlit.app/)

## Architecture

The system consists of three distinct components:
1.  **Modeling Pipeline (`model/`)**: Scikit-Learn pipeline handling data ingestion (UCI Repository), preprocessing, and Random Forest classification.
2.  **Inference API (`app/`)**: FastAPI-based REST service.
3.  **Frontend (`frontend/`)**: Streamlit interface for interactive risk querying.

## Local Development

### 1. Installation

Install dependencies from the requirements file.

```bash
pip install -r requirements.txt
```

### 2. Model Training

Execute the training script to fetch data and generate the serialized model artifact (`model/credit_risk_model.joblib`).

```bash
python model/train.py
```

### 3. API Initialization

Launch the inference server. The service binds to port 8000 by default.

```bash
python -m app.main
```

- **Health Check:** `GET http://localhost:8000/`
- **Swagger Documentation:** `GET http://localhost:8000/docs`

### 4. Frontend Launch

Start the Streamlit dashboard in a separate terminal.

```bash
streamlit run frontend/app.py
```
