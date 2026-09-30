import tensorflow as tf
from pathlib import Path

# =========================
# PATH
# =========================

MODEL_PATH = Path("models/eye_state_cnn.keras")
OUTPUT_PATH = Path("models/eye_state_cnn.tflite")


# =========================
# LOAD MODEL
# =========================

print("Loading Keras model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print()

model.summary()


# =========================
# CONVERT TO TFLITE
# =========================

print()
print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()


# =========================
# SAVE
# =========================

OUTPUT_PATH.write_bytes(
    tflite_model
)

print()
print("Conversion successful!")
print(
    f"Saved to: {OUTPUT_PATH}"
)

print(
    f"Model size: "
    f"{len(tflite_model) / 1024:.2f} KB"
)