from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI(title="Bati Bank Credit API")

# Load the model we saved earlier
model_path = os.path.join(os.path.dirname(__file__), '../../model.pkl')
model = pickle.load(open(model_path, 'rb'))

class CreditRequest(BaseModel):
    Amount: float
    Value: float
    hour: int
    day: int
    ProductCategory: int
    ChannelId: int
    PricingStrategy: int

@app.get("/")
def read_root():
    return {"status": "Credit Scoring API is online"}

@app.post("/predict")
def predict_risk(data: CreditRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "is_high_risk": int(prediction),
        "risk_probability": round(float(probability), 4),
        "recommendation": "Reject" if prediction == 1 else "Approve"
    }