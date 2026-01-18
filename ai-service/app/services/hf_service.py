from transformers import pipeline
from PIL import Image
import io

class HuggingFaceService:
    def __init__(self):
        # initializing Hugging Face image classification pipeline
        self.classifier = pipeline(
            task = "image-classification",
            model = "google/vit-base-patch16-224"
        )

    # takes image as bytes and returns top k predictions
    def classify_image(self, image_bytes: bytes, top_k: int = 3):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        predictions = self.classifier(image, top_k = top_k)

        result = []
        for pred in predictions:
            result.append({
                "label": pred["label"],
                "confidence": round(pred["score"], 4)
            })

        return result