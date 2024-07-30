from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Load the model
model = pickle.load(open('../models/talent_success_model.pkl', 'rb'))

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    data = pd.DataFrame([request.features])
    prediction = model.predict(data)[0]
    return {"prediction": prediction}

@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    predictions = model.predict(data)
    return {"predictions": predictions.tolist()}
