# TalentAI: Predicting Creative Success

## Project Overview

TalentAI is a machine learning platform designed to predict the success of creative projects or artists based on various features. This helps talent managers make data-driven decisions.

## Project Structure

- **notebook/**: Contains Jupyter Notebook for model evaluation.
- **src/**: Source code for model training, prediction, and FastAPI app.
- **data/**: Training and test datasets.
- **models/**: Trained model files.

## Setup Instructions

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the FastAPI app: `uvicorn src.app:app --reload`

## Model Details

The model is a Multi-layer Perceptron (MLP) trained on historical project data.

## Deployment

The application can be deployed using Docker. Build the Docker image and run it:

```bash
docker build -t talentai .
docker run -p 80:80 talentai
