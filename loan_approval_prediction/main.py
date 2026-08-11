from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Approval Prediction API",
    description="FastAPI backend for XGBoost Loan Approval Prediction",
    version="1.0.0"
)


# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load("xgboost_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# ============================================================
# 3. INPUT DATA SCHEMA
# ============================================================

class LoanApplication(BaseModel):

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str

    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float

    Property_Area: str


# ============================================================
# 4. HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Loan Approval Prediction API is running",
        "status": "success"
    }


# ============================================================
# 5. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_loan(application: LoanApplication):

    # Convert input to dictionary
    data = application.model_dump()

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame([data])

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = int(
        model.predict(input_processed)[0]
    )

    # Get probability
    probability = float(
        model.predict_proba(input_processed)[0][1]
    )

    # Convert prediction to readable result
    if prediction == 1:
        result = "Approved"
    else:
        result = "Not Approved"

    return {
        "prediction": prediction,
        "result": result,
        "approval_probability": round(
            probability * 100,
            2
        )
    }