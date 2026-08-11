import tensorflow as tf
import numpy as np
from pathlib import Path

# Echo: AI Auto EQ Model Generator
# This script generates a dummy genre classification model for Echo.
# Requires: tensorflow

def generate_model(output_path=None):
    # Input: 16000 samples (INT16) -> converted to FLOAT32 in model
    # Output: 5 genres (FLOAT32)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16000,), dtype=tf.int16),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        tf.keras.layers.Reshape((16000, 1)),
        tf.keras.layers.Conv1D(16, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    destination = Path(output_path or 'genre_model.tflite').expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as f:
        f.write(tflite_model)
    
    print(f"Successfully generated {destination}")

if __name__ == "__main__":
    try:
        generate_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure tensorflow is installed: pip install tensorflow")
