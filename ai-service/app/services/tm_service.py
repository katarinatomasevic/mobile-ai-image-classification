import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path

class TeachableMachineService:
    def __init__(self, model_path: str = "models/teachable_machine"):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = []
        self.image_size = (224, 224)
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        try:
            if self.model_path.suffix == '.h5':
                model_file = self.model_path
            else:
                model_file = self.model_path / "keras_model.h5"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found at {model_file}")

            self.model = tf.keras.models.load_model(str(model_file), compile=False)
            
            print(f"✓ Teachable Machine model loaded from {model_file}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Teachable Machine model: {str(e)}")
    
    def _load_labels(self):
        try:
            if self.model_path.suffix == '.h5':
                labels_file = self.model_path.parent / "labels.txt"
            else:
                labels_file = self.model_path / "labels.txt"
            
            if not labels_file.exists():
                raise FileNotFoundError(f"labels.txt not found at {labels_file}")
            
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip().split(maxsplit=1)[-1] 
                                   for line in f.readlines() if line.strip()]
            
            print(f"✓ Loaded {len(self.class_names)} classes: {self.class_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {str(e)}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def classify_image(self, image_bytes: bytes, top_k: int = 3):
        processed_image = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed_image, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        result = []
        for idx in top_indices:
            result.append({
                "label": self.class_names[idx],
                "confidence": round(float(predictions[idx]), 4)
            })
        return result
    
    def get_model_info(self):
        return {
            "model_type": "Teachable Machine (Keras)",
            "num_classes": len(self.class_names),
            "classes": self.class_names,
            "input_shape": str(self.model.input_shape) if self.model else None,
            "image_size": self.image_size
        }