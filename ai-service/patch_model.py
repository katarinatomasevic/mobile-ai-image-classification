import tensorflow as tf
import numpy as np
from pathlib import Path

def fix_teachable_machine_model():
    
    model_path = Path("models/teachable_machine/keras_model.h5")
    output_path = Path("models/teachable_machine/fixed_model")
    
    print("Loading original model...")
    # Load the model
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    print("Creating fixed version...")
    # Create a simple wrapper
    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = model(inputs, training=False)
    fixed_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save as SavedModel format (more stable)
    print(f"Saving to {output_path}...")
    fixed_model.save(str(output_path), save_format='tf')
    
    print("✓ Fixed model saved!")
    print("\nNow update tm_service.py to use this model:")
    print(f"  model_path='models/teachable_machine/fixed_model'")

if __name__ == "__main__":
    fix_teachable_machine_model()