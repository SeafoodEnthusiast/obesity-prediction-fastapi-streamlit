from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

# --- Load model & preprocessing ---
with open("model/best_model_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("model/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

# --- Initialize FastAPI ---
app = FastAPI()

# --- Input Data from user ---
class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# --- Prediction ---
@app.post("/predict")
def predict_obesity(data: ObesityInput):
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    print("Input dataframe:")
    print(df)

    try:
        # Add BMI features
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Scaling numerical fetures
        numeric_cols = scaler.feature_names_in_
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Prediction
        prediction = model.predict(df)
        result = target_encoder.inverse_transform(prediction)[0]

        return {"prediction": result}
    
    except Exception as e:
        print("Prediction Error:", e)
        return {"error": str(e)}

