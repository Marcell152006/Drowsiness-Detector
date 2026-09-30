from pathlib import Path

input_file = Path("eye_state_cnn_int8.tflite")
output_file = Path("eye_state_cnn_int8_models.h")

data = input_file.read_bytes()

with output_file.open("w") as f:
    f.write("#ifndef EYE_STATE_CNN_INT8_MODEL_H\n")
    f.write("#define EYE_STATE_CNN_INT8_MODEL_H\n\n")

    f.write("const unsigned char eye_state_cnn_int8_model[] = {\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        f.write("    ")
        f.write(", ".join(f"0x{b:02x}" for b in chunk))
        if i + 12 < len(data):
            f.write(",")
        f.write("\n")

    f.write("};\n\n")
    f.write(f"const unsigned int eye_state_cnn_int8_model_len = {len(data)};\n\n")
    f.write("#endif // EYE_STATE_CNN_INT8_MODEL_H\n")

print(f"Created: {output_file}")
print(f"Model size: {len(data)} bytes")