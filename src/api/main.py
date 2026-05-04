"""
FastAPI application for Accident Severity Prediction
Minimal deployment-ready API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import model classes
from models.catboost_model import CatBoostModel

app = FastAPI(
    title="Accident Severity Prediction API",
    description="Predict accident severity based on road and weather conditions",
    version="1.0.0"
)

# Pydantic models for input/output
class AccidentFeatures(BaseModel):
    Speed_limit: int
    Road_Type: str
    Weather_Conditions: str
    Light_Conditions: str
    Number_of_Vehicles: int
    Number_of_Casualties: int
    Day_of_Week: str
    Urban_or_Rural_Area: str
    
class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    confidence: float

# Global model instance
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        model = CatBoostModel()
        # Try to load a trained model - adjust path as needed
        model_path = PROJECT_ROOT / "models" / "trained_catboost.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model.model = pickle.load(f)
        else:
            # Create a basic model for demo purposes
            model.model = None
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Accident Severity Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_severity(features: AccidentFeatures):
    """Predict accident severity"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Create probability dictionary
        prob_dict = {
            "Fatal": float(probabilities[0]),
            "Serious": float(probabilities[1]), 
            "Slight": float(probabilities[2])
        }
        
        # Get confidence (max probability)
        confidence = max(prob_dict.values())
        
        return PredictionResponse(
            prediction=prediction,
            probabilities=prob_dict,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
