from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# CONFIG
# ============================================================

OPEN_IMAGE = Path(
    r"C:\Users\marce\Documents\ComVis Sleep\drowsiness_detection\dataset_split\test\Open_Eyes\s0001_02366_0_0_1_0_0_01.png"
)

CLOSED_IMAGE = Path(
    r"C:\Users\marce\Documents\ComVis Sleep\drowsiness_detection\dataset_split\test\Closed_Eyes\s0001_00002_0_0_0_0_0_01.png"
)

OUTPUT_FILE = Path(
    r"C:\Users\marce\Documents\Arduino\esp32_int8_test\eye_test_images.h"
)

IMAGE_SIZE = (64, 64)


# ============================================================
# IMAGE → C ARRAY
# ============================================================

def image_to_array(image_path):
    print(f"Loading: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.uint8)

    print(f"  Shape: {array.shape}")
    print(f"  Min:   {array.min()}")
    print(f"  Max:   {array.max()}")

    return array


def array_to_c(array, variable_name):
    flat = array.flatten()

    lines = []

    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        line = ", ".join(str(int(x)) for x in chunk)
        lines.append("    " + line)

    return (
        f"const uint8_t {variable_name}[{len(flat)}] = {{\n"
        + ",\n".join(lines)
        + "\n};\n"
    )


# ============================================================
# MAIN
# ============================================================

open_array = image_to_array(OPEN_IMAGE)
closed_array = image_to_array(CLOSED_IMAGE)

assert open_array.shape == (64, 64, 3)
assert closed_array.shape == (64, 64, 3)

header = """#ifndef EYE_TEST_IMAGES_H
#define EYE_TEST_IMAGES_H

#include <stdint.h>

"""

header += array_to_c(
    open_array,
    "open_eye_image"
)

header += "\n"

header += array_to_c(
    closed_array,
    "closed_eye_image"
)

header += """

#endif
"""

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(header, encoding="utf-8")

print()
print("========================================")
print("DONE")
print("========================================")
print(f"Output: {OUTPUT_FILE}")
print(f"Open image bytes:   {open_array.size}")
print(f"Closed image bytes: {closed_array.size}")