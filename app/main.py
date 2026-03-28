from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the model at startup
model = joblib.load("app/model.pkl")

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is Up!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/ready")
async def readiness_check():
    if model_is_loaded: # Check if your .pkl or .h5 file is ready
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "loading_model"})

@app.post("/predict")
def predict(data: list):
    # Data should be a list of 8 features (MedInc, HouseAge, etc.)
    prediction = model.predict([data])
    return {"estimated_value": float(prediction[0])}
