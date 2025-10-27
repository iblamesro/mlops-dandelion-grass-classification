"""
FastAPI application for plant classification
"""

import io
import time
from typing import Optional
from datetime import datetime

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from src.config import settings
from src.training.model import load_model
from src.utils.helpers import preprocess_image, image_to_tensor


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Dandelion vs Grass classification"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Global model variable
model = None
device = None
class_names = ["dandelion", "grass"]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    prediction_time: float
    timestamp: str


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when API starts"""
    global model, device
    
    logger.info("Loading model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        model_path = settings.MODELS_DIR / f"{settings.MODEL_NAME}_best.pth"
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, creating dummy model")
            from src.training.model import create_model
            model = create_model(pretrained=True)
            model.to(device)
        else:
            model = load_model(str(model_path), device)
        
        model.eval()
        logger.success("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Classification API",
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    
    Args:
        file: Image file to classify
        
    Returns:
        Prediction response with class and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Preprocess image
        image = preprocess_image(image, target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        
        # Convert to tensor
        img_tensor = image_to_tensor(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Normalize
        normalize = torch.nn.functional.normalize
        img_tensor = img_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        prediction_time = time.time() - start_time
        
        logger.info(f"Prediction: {predicted_class} ({confidence_score:.4f}) in {prediction_time:.3f}s")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence_score,
            prediction_time=prediction_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "classes": class_names,
        "image_size": settings.IMAGE_SIZE,
        "device": str(device),
        "model_type": "ResNet18"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
