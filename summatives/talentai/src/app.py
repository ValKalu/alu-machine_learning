from fastapi import FastAPI, UploadFile, File
from src.prediction import predict
from src.model import model_training  # Assuming you have a retraining function in model.py

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to TalentAI"}

@app.post("/predict")
async def make_prediction(features: list):
    prediction = predict(features)
    return {"prediction": prediction}

@app.post("/retrain")
async def retrain_model():
    model_training()
    return {"message": "Model retrained successfully"}
