from PIL import Image
import io
from pathlib import Path
import os

class HuggingFaceService:
    def __init__(self):
        os.environ['USE_TF'] = '0'
        os.environ['USE_TORCH'] = '1'
        
        from transformers import pipeline
        
        local_model_path = "hf_models"
        
        if Path(local_model_path).exists():
            print(f"✓ Loading Hugging Face model from local cache: {local_model_path}")
            model_source = local_model_path
        else:
            print("⚠ Local model not found, will attempt to download from Hugging Face")
            model_source = "google/vit-base-patch16-224"
        
        try:
            self.classifier = pipeline(
                task="image-classification",
                model=model_source,
                device=-1,  
                framework="pt"  
            )
            print("✓ Hugging Face model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Hugging Face model: {str(e)}")
            raise

    def classify_image(self, image_bytes: bytes, top_k: int = 3):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        predictions = self.classifier(image, top_k=top_k)

        result = []
        for pred in predictions:
            result.append({
                "label": pred["label"],
                "confidence": round(pred["score"], 4)
            })

        return result