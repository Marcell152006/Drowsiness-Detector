import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/eye_state_cnn.tflite")

OPEN_IMAGE = Path(
    "dataset/train/Open_Eyes/s0001_02336_0_0_1_0_0_01.png"
)

CLOSED_IMAGE = Path(
    "dataset/train/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print()
print("Input shape :", input_details[0]["shape"])
print("Input dtype :", input_details[0]["dtype"])

print()
print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (64, 64)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return float(output[0][0])


# ============================================================
# OPEN IMAGE
# ============================================================

print()
print("==============================")
print("OPEN EYES TEST")
print("==============================")

if not OPEN_IMAGE.exists():

    print("ERROR: Open image not found:")
    print(OPEN_IMAGE)

else:

    prediction = predict(
        OPEN_IMAGE
    )

    print("File       :", OPEN_IMAGE)
    print(f"Raw output : {prediction:.6f}")


# ============================================================
# CLOSED IMAGE
# ============================================================

print()
print("==============================")
print("CLOSED EYES TEST")
print("==============================")

if not CLOSED_IMAGE.exists():

    print("ERROR: Closed image not found:")
    print(CLOSED_IMAGE)

else:

    prediction = predict(
        CLOSED_IMAGE
    )

    print("File       :", CLOSED_IMAGE)
    print(f"Raw output : {prediction:.6f}")


print()
print("==============================")
print("TEST FINISHED")
print("==============================")