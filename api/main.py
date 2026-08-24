"""
Phase 10: FastAPI Backend Service
---------------------------------
This is the REST API that exposes our trained PyTorch model to the outside world.
It uses FastAPI for extreme performance and Pydantic for strict input validation.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from src.models.network import NIDSNeuralNet
# Import preprocessor to make it available for joblib unpickling
from src.data.preprocess import NIDSPreprocessor
import __main__
__main__.NIDSPreprocessor = NIDSPreprocessor

# Global variables to hold our model and preprocessor in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs once when the API starts up.
    We load the heavy PyTorch model and Preprocessor here so we don't
    have to reload them from disk on every single HTTP request.
    """
    print("[INFO] Starting up API and loading models into memory...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    config_path = os.path.join(models_dir, "model_config.json")
    model_path = os.path.join(models_dir, "nids_model.pt")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")

    if not all(os.path.exists(p) for p in [config_path, model_path, preprocessor_path]):
        raise RuntimeError("Missing model artifacts! Please run train.py first.")

    # 1. Load Preprocessor
    preprocessor = joblib.load(preprocessor_path)
    ml_models["preprocessor"] = preprocessor

    # 2. Load PyTorch Model
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cpu") # Force CPU for API inference to ensure compatibility
    model = NIDSNeuralNet(input_dim=config["input_dim"], num_classes=config["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode (disables dropout)
    
    ml_models["model"] = model
    ml_models["classes"] = list(preprocessor.label_encoder.classes_)
    
    print(f"[SUCCESS] Model loaded with {config['num_classes']} target classes.")
    
    yield # API is now running and accepting requests
    
    # Cleanup on shutdown
    print("[INFO] Shutting down API and clearing memory...")
    ml_models.clear()


# Initialize FastAPI application
app = FastAPI(
    title="AI-NIDS Prediction API",
    description="Real-time Network Intrusion Detection using PyTorch",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS so our React frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC SCHEMAS (Input Validation)
# ==========================================
class NetworkFlowInput(BaseModel):
    # Using a generic Dict to accept the 40+ features of UNSW-NB15.
    # In a rigid enterprise setting, you would explicitly define all 45 fields here
    # (e.g., dur: float, proto: str, spkts: int, etc.)
    features: Dict[str, Any]

class BatchNetworkFlowInput(BaseModel):
    flows: List[NetworkFlowInput]


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Root endpoint providing links to documentation."""
    return {
        "message": "Welcome to the AI-Powered Network Intrusion Detection System API",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model/info"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": "model" in ml_models}

@app.get("/model/info")
async def model_info():
    """Returns metadata about the currently loaded model."""
    if "classes" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "framework": "PyTorch",
        "supported_classes": ml_models["classes"],
        "num_classes": len(ml_models["classes"])
    }

@app.post("/predict")
async def predict_flow(flow: NetworkFlowInput):
    """
    Predicts whether a single network flow is benign or malicious.
    """
    try:
        # 1. Convert JSON payload to Pandas DataFrame
        df = pd.DataFrame([flow.features])
        
        # 2. Preprocess data (same as training pipeline!)
        preprocessor = ml_models["preprocessor"]
        X_proc, _ = preprocessor.transform(df, is_test=False)
        
        # 3. Convert to PyTorch Tensor
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        
        # 4. Neural Network Forward Pass
        model = ml_models["model"]
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[0] # Get probabilities for the first (and only) item
            
        # 5. Extract Prediction
        confidence_score, pred_idx = torch.max(probs, dim=0)
        predicted_class = ml_models["classes"][pred_idx.item()]
        
        # Format the class probability distribution
        prob_dist = {ml_models["classes"][i]: round(probs[i].item(), 4) for i in range(len(ml_models["classes"]))}
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence_score.item(), 4),
            "class_probabilities": prob_dist,
            "is_anomaly": predicted_class != "Normal"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(batch: BatchNetworkFlowInput):
    """
    Predicts multiple network flows in a single API call.
    """
    try:
        # Convert all JSON payloads into a single DataFrame
        df = pd.DataFrame([flow.features for flow in batch.flows])
        
        preprocessor = ml_models["preprocessor"]
        X_proc, _ = preprocessor.transform(df, is_test=False)
        
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        model = ml_models["model"]
        
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            
        results = []
        for i in range(len(probs)):
            confidence_score, pred_idx = torch.max(probs[i], dim=0)
            predicted_class = ml_models["classes"][pred_idx.item()]
            prob_dist = {ml_models["classes"][j]: round(probs[i][j].item(), 4) for j in range(len(ml_models["classes"]))}
            
            results.append({
                "flow_index": i,
                "prediction": predicted_class,
                "confidence": round(confidence_score.item(), 4),
                "class_probabilities": prob_dist,
                "is_anomaly": predicted_class != "Normal"
            })
            
        return {"batch_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

# To run locally: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
