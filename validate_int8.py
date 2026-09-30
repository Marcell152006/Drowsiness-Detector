import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/eye_state_cnn_int8.tflite"
)

DATASET_PATH = Path(
    "dataset_split/test"
)

IMAGE_SIZE = (64, 64)

THRESHOLD = 0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


print()
print("==============================")
print("MODEL INFORMATION")
print("==============================")

print(
    "Input shape :",
    input_details[0]["shape"]
)

print(
    "Input dtype :",
    input_details[0]["dtype"]
)

print(
    "Input scale :",
    input_scale
)

print(
    "Input zero point:",
    input_zero_point
)

print()

print(
    "Output shape:",
    output_details[0]["shape"]
)

print(
    "Output dtype:",
    output_details[0]["dtype"]
)

print(
    "Output scale:",
    output_scale
)

print(
    "Output zero point:",
    output_zero_point
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

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

    # Same preprocessing as training
    image = image / 255.0

    # Convert FLOAT32 → INT8
    image = (
        image / input_scale
        + input_zero_point
    )

    image = np.round(
        image
    )

    image = np.clip(
        image,
        -128,
        127
    )

    image = image.astype(
        np.int8
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    interpreter.set_tensor(
        input_index,
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    raw_output = int(
        output[0][0]
    )

    # Dequantize output
    probability = (
        raw_output - output_zero_point
    ) * output_scale

    # >= 0.5 = OPEN
    # < 0.5  = CLOSED

    if probability >= 0.5:
        predicted_label = 1
    else:
        predicted_label = 0

    return predicted_label, probability, raw_output


# ============================================================
# DATASET
# ============================================================

open_folder = DATASET_PATH / "Open_Eyes"
closed_folder = DATASET_PATH / "Closed_Eyes"

open_images = list(
    open_folder.glob("*.png")
)

closed_images = list(
    closed_folder.glob("*.png")
)


print()
print("==============================")
print("DATASET")
print("==============================")

print(
    "Open Eyes   :",
    len(open_images)
)

print(
    "Closed Eyes :",
    len(closed_images)
)

print(
    "Total       :",
    len(open_images) + len(closed_images)
)


# ============================================================
# RUN TEST
# ============================================================

true_labels = []
predicted_labels = []

wrong_predictions = []


print()
print("==============================")
print("RUNNING INT8 INFERENCE")
print("==============================")


# ------------------------------------------------------------
# OPEN EYES
# ------------------------------------------------------------

for image_path in open_images:

    true_label = 1

    try:

        predicted_label, probability, raw_output = predict(
            image_path
        )

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

        if predicted_label != true_label:

            wrong_predictions.append(
                (
                    image_path,
                    "OPEN",
                    "CLOSED",
                    probability,
                    raw_output
                )
            )

    except Exception as e:

        print(
            "ERROR:",
            image_path
        )

        print(e)


# ------------------------------------------------------------
# CLOSED EYES
# ------------------------------------------------------------

for image_path in closed_images:

    true_label = 0

    try:

        predicted_label, probability, raw_output = predict(
            image_path
        )

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

        if predicted_label != true_label:

            wrong_predictions.append(
                (
                    image_path,
                    "CLOSED",
                    "OPEN",
                    probability,
                    raw_output
                )
            )

    except Exception as e:

        print(
            "ERROR:",
            image_path
        )

        print(e)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

correct = sum(
    t == p
    for t, p in zip(
        true_labels,
        predicted_labels
    )
)

wrong = len(
    wrong_predictions
)


print()
print("==============================")
print("INT8 RESULTS")
print("==============================")

print(
    "Total tested :",
    len(true_labels)
)

print(
    "Correct      :",
    correct
)

print(
    "Wrong        :",
    wrong
)

print(
    f"Accuracy     : {accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")

print()
print(
    "                 Predicted"
)

print(
    "                 CLOSED   OPEN"
)

print(
    f"Actual CLOSED   "
    f"{cm[0][0]:6d} "
    f"{cm[0][1]:7d}"
)

print(
    f"Actual OPEN     "
    f"{cm[1][0]:6d} "
    f"{cm[1][1]:7d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "CLOSED",
            "OPEN"
        ],
        digits=4
    )
)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print()
print("==============================")
print("WRONG PREDICTIONS")
print("==============================")

if len(wrong_predictions) == 0:

    print(
        "No wrong predictions!"
    )

else:

    print(
        f"Total wrong: "
        f"{len(wrong_predictions)}"
    )

    print()

    for item in wrong_predictions[:20]:

        (
            image_path,
            actual,
            predicted,
            probability,
            raw_output
        ) = item

        print(
            "File      :",
            image_path
        )

        print(
            "Actual    :",
            actual
        )

        print(
            "Predicted :",
            predicted
        )

        print(
            f"Probability: "
            f"{probability:.6f}"
        )

        print(
            "Raw INT8 output:",
            raw_output
        )

        print(
            "------------------------------"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("==============================")
print("VALIDATION FINISHED")
print("==============================")