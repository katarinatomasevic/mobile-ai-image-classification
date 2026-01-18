from PIL import Image
import io
import numpy as np

class ImagePreprocessorService:
    def __init__(self):
        self.tm_image_size = (224, 224)

    # prepares image for Teachable Machine (Keras) model
    def preprocess_for_hf(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.tm_image_size)
        
        image_array = np.array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis = 0)

        return image_array