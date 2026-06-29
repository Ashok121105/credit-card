# Credit Card Approval Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a credit card application will be **Approved** or **Rejected** based on applicant financial and demographic information.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Screenshots

| Home Page | Prediction Page | Dashboard |
|-----------|-----------------|-----------|
| ![Home](screenshots/home_page.png) | ![Predict](screenshots/predict_page.png) | ![Dashboard](screenshots/dashboard_page.png) |

> **Note:** Run the app and capture screenshots to replace placeholders in `screenshots/`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Model Comparison](#model-comparison)
- [System Diagrams](#system-diagrams)
- [Deployment](#deployment)
- [License](#license)

---

## Project Overview

This project implements a complete ML pipeline:

1. **Data Preprocessing** — duplicates, missing values, encoding, scaling, outliers, feature engineering
2. **Model Training** — Logistic Regression, Decision Tree, Random Forest, XGBoost
3. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
4. **Web Application** — Flask app with prediction form and analytics dashboard

---

## Features

- 15 input features matching real credit application forms
- Automatic best model selection (F1 score)
- Pickle model persistence for production deployment
- Responsive Bootstrap 5 UI
- JavaScript form validation
- REST API for programmatic predictions
- Interactive analytics dashboard with charts

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost |
| Web Framework | Flask |
| Frontend | Bootstrap 5, HTML, CSS, JavaScript |
| Visualization | Matplotlib, Seaborn |
| Deployment | Gunicorn, Render, Railway |

---

## Project Structure

```
CreditCardApproval/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── model.pkl               # Trained model artifact
├── dataset/
│   └── credit_card.csv     # Credit card approval dataset
├── notebooks/
│   └── model_training.ipynb
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── dashboard.html
├── static/
│   ├── css/style.css
│   ├── js/validation.js
│   └── images/             # Evaluation charts
├── models/
│   └── train_model.py      # Training pipeline script
├── utils/
│   ├── preprocess.py       # Data preprocessing
│   └── helper.py           # Evaluation & visualization
└── screenshots/            # App screenshots
```

---

## Installation

```bash
# Clone or navigate to project
cd CreditCardApproval

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python models/train_model.py
```

This generates `model.pkl`, `dataset/credit_card.csv`, and charts in `static/images/`.

### 2. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Production Server

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Model Training

The training script (`models/train_model.py`) performs:

| Step | Description |
|------|-------------|
| Data Loading | Load or generate 1,500+ sample records |
| Duplicate Removal | Remove duplicate rows |
| Missing Values | Median (numeric), mode (categorical) |
| Outlier Treatment | IQR capping method |
| Feature Engineering | Income consistency, loan ratios, credit risk |
| Encoding | Label encoding for categorical features |
| Scaling | StandardScaler for numeric features |
| Train-Test Split | 80/20 stratified split |
| Model Training | 4 classifiers trained and compared |
| Evaluation | Full metrics + visualization charts |
| Model Save | Best model saved to `model.pkl` |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | GET/POST | Prediction form |
| `/dashboard` | GET | Analytics dashboard |
| `/api/metrics` | GET | JSON model metrics |
| `/api/predict` | POST | REST prediction API |

### Example API Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 35,
    "annual_income": 75000,
    "employment_status": "Employed",
    "years_of_employment": 8,
    "education_level": "Bachelor",
    "marital_status": "Married",
    "number_of_dependents": 2,
    "existing_loan_amount": 10000,
    "monthly_income": 6000,
    "credit_score": 720,
    "loan_status": "None",
    "credit_history": "Good",
    "number_of_credit_inquiries": 1,
    "debt_to_income_ratio": 0.25
  }'
```

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.79 | 0.78 | 0.31 | 0.44 | 0.70 |
| Decision Tree | 0.71 | 0.44 | 0.31 | 0.36 | 0.63 |
| Random Forest | 0.79 | 0.84 | 0.26 | 0.40 | 0.75 |
| **XGBoost** | **0.79** | **0.77** | **0.33** | **0.47** | **0.76** |

> Best model selected automatically based on **F1 Score** (balances precision and recall).

---

## System Diagrams

### System Architecture Diagram

```mermaid
graph TB
    subgraph Client Layer
        A[Web Browser]
        B[REST API Client]
    end

    subgraph Application Layer
        C[Flask Web App<br/>app.py]
        D[Templates<br/>HTML/CSS/JS]
    end

    subgraph ML Layer
        E[Preprocessor<br/>utils/preprocess.py]
        F[ML Model<br/>model.pkl]
        G[Training Pipeline<br/>models/train_model.py]
    end

    subgraph Data Layer
        H[Dataset<br/>credit_card.csv]
        I[Metrics JSON<br/>metrics.json]
        J[Charts<br/>static/images/]
    end

    A --> C
    B --> C
    C --> D
    C --> E
    E --> F
    F --> C
    G --> H
    G --> F
    G --> I
    G --> J
    C --> I
    C --> J
```

### Workflow Diagram

```mermaid
flowchart LR
    A[Raw Dataset] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Encoding & Scaling]
    D --> E[Train-Test Split]
    E --> F[Train Models]
    F --> G[Evaluate Models]
    G --> H{Best Model?}
    H --> I[Save model.pkl]
    I --> J[Flask Web App]
    J --> K[User Prediction]
    K --> L[Preprocess Input]
    L --> M[Model Inference]
    M --> N[Display Result]
```

### Use Case Diagram

```mermaid
graph LR
    User((User))
    Admin((Admin))

    User --> UC1[View Home Page]
    User --> UC2[Submit Prediction]
    User --> UC3[View Dashboard]
    User --> UC4[Reset Form]

    Admin --> UC5[Train Models]
    Admin --> UC6[View Metrics]
    Admin --> UC7[Deploy Application]

    UC2 --> UC8[Validate Input]
    UC8 --> UC9[Get Prediction Result]
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Flask App
    participant P as Preprocessor
    participant M as ML Model

    U->>F: Submit prediction form
    F->>F: Validate form data
    F->>P: encode_input(data)
    P->>P: handle_missing, engineer_features
    P->>P: transform_encode, scale
    P->>M: predict(X)
    M-->>P: prediction + probability
    P-->>F: decoded result
    F-->>U: Approved/Rejected + probability
```

### Data Flow Diagram - Level 0

```mermaid
flowchart LR
    User[User] -->|Applicant Data| System[Credit Card<br/>Approval System]
    System -->|Prediction Result| User
    Admin[Admin] -->|Training Data| System
    System -->|Metrics & Charts| Admin
```

### Data Flow Diagram - Level 1

```mermaid
flowchart TB
    User[User] -->|Form Input| P1[1.0 Prediction<br/>Processing]
    P1 -->|Features| P2[2.0 Model<br/>Inference]
    P2 -->|Result| User

    Admin[Admin] -->|CSV Data| D1[3.0 Data<br/>Preprocessing]
    D1 -->|Clean Data| D2[4.0 Model<br/>Training]
    D2 -->|model.pkl| D3[5.0 Evaluation<br/>& Storage]
    D3 -->|Charts| Admin

    D3 -->|model.pkl| P2
```

### ER Diagram

```mermaid
erDiagram
    APPLICANT {
        int applicant_id PK
        string gender
        int age
        float annual_income
        string employment_status
        float years_of_employment
        string education_level
        string marital_status
        int number_of_dependents
        float existing_loan_amount
        float monthly_income
        int credit_score
        string loan_status
        string credit_history
        int number_of_credit_inquiries
        float debt_to_income_ratio
    }

    PREDICTION {
        int prediction_id PK
        int applicant_id FK
        string approval_status
        float probability
        string model_name
        datetime timestamp
    }

    MODEL {
        int model_id PK
        string model_name
        float accuracy
        float precision
        float recall
        float f1_score
        float roc_auc
        datetime trained_at
    }

    APPLICANT ||--o{ PREDICTION : generates
    MODEL ||--o{ PREDICTION : produces
```

---

## Deployment

### IBM Watson Machine Learning

1. Create an IBM Cloud account and Watson Machine Learning service
2. Upload `model.pkl` to Watson Model Registry
3. Create a deployment endpoint
4. Update Flask app to call Watson API or deploy Flask on IBM Cloud Code Engine:

```bash
# Install IBM CLI and login
ibmcloud login
ibmcloud ce project create --name credit-card-ml
ibmcloud ce app create --name credit-card-app \
  --image python:3.11 \
  --build-source . \
  --cmd "gunicorn app:app --bind 0.0.0.0:8080"
```

### Streamlit Community Cloud

Create `streamlit_app.py` as an alternative UI:

```python
import pickle
import streamlit as st
from utils.preprocess import DataPreprocessor

artifact = pickle.load(open("model.pkl", "rb"))
model = artifact["model"]
preprocessor = artifact["preprocessor"]

st.title("Credit Card Approval Prediction")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 30)
# ... add other fields
if st.button("Predict"):
    data = {"gender": gender, "age": age, ...}
    X = preprocessor.transform(preprocessor.encode_input(data))
    result = preprocessor.decode_target(model.predict(X)[0])
    st.success(f"Result: {result}")
```

Deploy via [share.streamlit.io](https://share.streamlit.io) by connecting your GitHub repo.

### Render

1. Push project to GitHub
2. Create new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt && python models/train_model.py`
4. Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Deploy

Create `render.yaml`:

```yaml
services:
  - type: web
    name: credit-card-approval
    env: python
    buildCommand: pip install -r requirements.txt && python models/train_model.py
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
```

### Railway

1. Push project to GitHub
2. Create new project on [railway.app](https://railway.app)
3. Connect GitHub repository
4. Railway auto-detects Python; set start command:

```
gunicorn app:app --bind 0.0.0.0:$PORT
```

5. Add environment variable `PORT` if needed
6. Deploy

Create `Procfile`:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## License

This project is open source and available under the MIT License.

---

## Author

Built as an end-to-end Machine Learning portfolio project demonstrating data science and full-stack Python development best practices.
