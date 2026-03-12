from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the model at startup
model = joblib.load("app/model.pkl")

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is Up!"}

@app.post("/predict")
def predict(data: list):
    # Data should be a list of 8 features (MedInc, HouseAge, etc.)
    prediction = model.predict([data])
    return {"estimated_value": float(prediction[0])}