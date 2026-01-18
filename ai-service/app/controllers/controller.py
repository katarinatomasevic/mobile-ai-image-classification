from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.hf_service import HuggingFaceService

router = APIRouter()

hf_service = HuggingFaceService()

# image classification using Hugging Face model
@router.post("/classify/hf")
async def classify_with_huggingface(
    image: UploadFile = File(...)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File is not an image."
        )

    image_bytes = await image.read()

    predictions = hf_service.classify_image(image_bytes)

    return {
        "model": "huggingface",
        "filename": image.filename,
        "predictions": predictions
    }


@router.post("/classify/tm")
async def classify_with_teachable_machine(
    image: UploadFile = File(...)
):
    raise HTTPException(
        status_code=501,
        detail="Teachable Machine endpoint not implemented."
    )
