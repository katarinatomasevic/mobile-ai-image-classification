from fastapi import FastAPI
from app.controllers.controller import router as api_router

app = FastAPI(
    title="AI Image Classification Service",
    description="Service for image classification (Hugging Face i Teachable Machine)",
    version="1.0.0"
)

app.include_router(api_router)


@app.get("/")
def health_check():
    return {"status": "OK"}