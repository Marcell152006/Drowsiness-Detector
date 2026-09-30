import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/eye_state_cnn.keras"
)

OUTPUT_PATH = Path(
    "models/eye_state_cnn_int8.tflite"
)

DATASET_PATH = Path(
    "dataset/train"
)

IMAGE_SIZE = (64, 64)

CALIBRATION_IMAGES_PER_CLASS = 150


# ============================================================
# CHECK FILES
# ============================================================

print("Checking files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

print("Model found  :", MODEL_PATH)
print("Dataset found:", DATASET_PATH)


# ============================================================
# LOAD KERAS MODEL
# ============================================================

print()
print("Loading Keras model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD CALIBRATION IMAGES
# ============================================================

open_folder = DATASET_PATH / "Open_Eyes"
closed_folder = DATASET_PATH / "Closed_Eyes"

open_images = sorted(
    open_folder.glob("*.png")
)

closed_images = sorted(
    closed_folder.glob("*.png")
)


if len(open_images) == 0:
    raise FileNotFoundError(
        f"No Open_Eyes images found in {open_folder}"
    )

if len(closed_images) == 0:
    raise FileNotFoundError(
        f"No Closed_Eyes images found in {closed_folder}"
    )


# Select calibration images
open_calibration = open_images[
    :CALIBRATION_IMAGES_PER_CLASS
]

closed_calibration = closed_images[
    :CALIBRATION_IMAGES_PER_CLASS
]

calibration_images = (
    open_calibration +
    closed_calibration
)


print()
print("==============================")
print("CALIBRATION DATA")
print("==============================")

print(
    "Open Eyes   :",
    len(open_calibration)
)

print(
    "Closed Eyes :",
    len(closed_calibration)
)

print(
    "Total       :",
    len(calibration_images)
)


# ============================================================
# REPRESENTATIVE DATASET
# ============================================================

def representative_dataset():

    for image_path in calibration_images:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        image = np.array(
            image,
            dtype=np.float32
        )

        # Same preprocessing as model training
        image = image / 255.0

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        yield [image]


# ============================================================
# CREATE CONVERTER
# ============================================================

print()
print("==============================")
print("CREATING TFLITE CONVERTER")
print("==============================")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)


# ============================================================
# ENABLE QUANTIZATION
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)


# ============================================================
# FORCE FULL INTEGER QUANTIZATION
# ============================================================

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# ============================================================
# CONVERT
# ============================================================

print()
print("==============================")
print("CONVERTING TO INT8")
print("==============================")

tflite_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.write_bytes(
    tflite_model
)


# ============================================================
# RESULTS
# ============================================================

model_size_kb = (
    len(tflite_model) / 1024
)

print()
print("==============================")
print("QUANTIZATION SUCCESSFUL")
print("==============================")

print(
    "Saved to:"
)

print(
    OUTPUT_PATH
)

print()

print(
    f"Model size: "
    f"{model_size_kb:.2f} KB"
)

print()
print("==============================")
print("DONE")
print("==============================")