from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.hf_service import HuggingFaceService
from app.services.tm_service import TeachableMachineService

router = APIRouter()

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
async def classify_with_huggingface(
    image: UploadFile = File(...),
    top_k: int = 3
):

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File is not an image."
        )

    image_bytes = await image.read()
    predictions = hf_service.classify_image(image_bytes, top_k=top_k)

    return {
        "model": "huggingface",
        "model_name": "google/vit-base-patch16-224",
        "filename": image.filename,
        "predictions": predictions
    }


# Image classification using Teachable Machine model
@router.post("/classify/tm")
async def classify_with_teachable_machine(
    image: UploadFile = File(...),
    top_k: int = 3
):

    if not tm_enabled:
        raise HTTPException(
            status_code=503,
            detail="Teachable Machine model not loaded. Please check model files."
        )
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File is not an image."
        )

    try:
        image_bytes = await image.read()
        predictions = tm_service.classify_image(image_bytes, top_k=top_k)

        return {
            "model": "teachable_machine",
            "filename": image.filename,
            "predictions": predictions,
            "classes": tm_service.class_names
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )



@router.get("/models/tm/info")
async def get_tm_model_info():

    if not tm_enabled:
        raise HTTPException(
            status_code=503,
            detail="Teachable Machine model not loaded."
        )
    
    return tm_service.get_model_info()