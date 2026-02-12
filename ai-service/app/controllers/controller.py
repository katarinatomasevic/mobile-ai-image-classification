from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.services.hf_service import HuggingFaceService
from app.services.tm_service import TeachableMachineService

import logging

router = APIRouter()

logger = logging.getLogger(__name__)

hf_service = HuggingFaceService()


try:
    tm_service = TeachableMachineService(model_path="models/teachable_machine")
    tm_enabled = True
except Exception as e:
    print(f"⚠ Teachable Machine model not loaded: {str(e)}")
    print("  Place keras_model.h5 and labels.txt in models/teachable_machine/")
    tm_service = None
    tm_enabled = False


# Health check with model status
@router.get("/status")
async def get_status():
    return {
        "status": "OK",
        "models": {
            "huggingface": {
                "enabled": True,
                "model": "google/vit-base-patch16-224"
            },
            "teachable_machine": {
                "enabled": tm_enabled,
                "info": tm_service.get_model_info() if tm_enabled else None
            }
        }
    }


# Image classification using Hugging Face model
@router.post("/classify/hf")
async def classify_with_huggingface(request: Request):
    #logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    try:
        image_bytes = await request.body()
        logger.info(f"Read {len(image_bytes)} bytes")
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image not found.")
        
        predictions = hf_service.classify_image(image_bytes)
        logger.info(f"Predictions: {predictions}")
        
        return {
            "model": "huggingface",
            "model_name": "google/vit-base-patch16-224",
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Image classification using Teachable Machine model
@router.post("/classify/tm")
async def classify_with_teachable_machine(request: Request):
    #logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    if not tm_enabled:
        raise HTTPException(status_code=503, detail="TM model not loaded")
    
    try:
        image_bytes = await request.body()
        logger.info(f"Read {len(image_bytes)} bytes")
        
        predictions = tm_service.classify_image(image_bytes)
        logger.info(f"Predictions: {predictions}")
        
        return {
            "model": "teachable_machine",
            "predictions": predictions,
            "classes": tm_service.class_names
        }
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/models/tm/info")
async def get_tm_model_info():

    if not tm_enabled:
        raise HTTPException(
            status_code=503,
            detail="Teachable Machine model not loaded."
        )
    
    return tm_service.get_model_info()