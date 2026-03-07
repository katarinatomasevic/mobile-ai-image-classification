#!/usr/bin/env python3

from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path

MODEL_NAME = "google/vit-base-patch16-224"
CACHE_DIR = "./hf_models"

def download_model():

    print(f"Downloading model: {MODEL_NAME}")
    print(f"Cache directory: {CACHE_DIR}")

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    
    processor.save_pretrained(CACHE_DIR)
    model.save_pretrained(CACHE_DIR)
    
    print(f"✓ Model downloaded successfully to {CACHE_DIR}")
    print(f"✓ You can now build the Docker image")

if __name__ == "__main__":
    download_model()