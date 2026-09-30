import tensorflow as tf

MODEL_PATH = r"C:\Users\marce\Documents\ComVis Sleep\drowsiness_detection\models\eye_state_cnn_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

ops = interpreter._get_ops_details()

print("=" * 60)
print("TFLITE MODEL OPERATORS")
print("=" * 60)

unique_ops = []

for i, op in enumerate(ops):
    name = op["op_name"]

    if name not in unique_ops:
        unique_ops.append(name)

    print(
        f"{i:02d} | "
        f"{name}"
    )

print()
print("=" * 60)
print("UNIQUE OPERATORS")
print("=" * 60)

for i, name in enumerate(unique_ops):
    print(f"{i+1:02d}. {name}")

print()
print(f"Total operator nodes : {len(ops)}")
print(f"Unique operators     : {len(unique_ops)}")