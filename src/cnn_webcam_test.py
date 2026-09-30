import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "eye_state_cnn.keras"

IMG_SIZE = 64

# 0 = Closed_Eyes
# 1 = Open_Eyes
CLASS_NAMES = ["Closed", "Open"]

# MediaPipe eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CNN model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_eye_crop(frame, landmarks, eye_indices):
    """
    Mengambil area mata berdasarkan landmark MediaPipe.
    """

    h, w, _ = frame.shape

    points = []

    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # Tambahkan padding agar crop tidak terlalu ketat
    eye_width = x_max - x_min
    eye_height = y_max - y_min

    padding_x = int(eye_width * 0.5)
    padding_y = int(eye_height * 1.0)

    x_min = max(0, x_min - padding_x)
    x_max = min(w, x_max + padding_x)

    y_min = max(0, y_min - padding_y)
    y_max = min(h, y_max + padding_y)

    crop = frame[y_min:y_max, x_min:x_max]

    return crop, (x_min, y_min, x_max, y_max)


def predict_eye(crop):
    """
    Preprocess crop lalu prediksi menggunakan CNN.
    """

    if crop.size == 0:
        return None, None

    # OpenCV BGR → RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Resize ke ukuran input CNN
    crop_resized = cv2.resize(
        crop_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalisasi 0-255 → 0-1
    crop_normalized = crop_resized.astype(np.float32) / 255.0

    # Tambahkan batch dimension
    input_image = np.expand_dims(
        crop_normalized,
        axis=0
    )

    # Prediction
    probability = float(model.predict(input_image, verbose=0)[0][0])

    predicted_class = 1 if probability >= 0.5 else 0

    return CLASS_NAMES[predicted_class], probability


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()


print("Webcam started.")
print("Tekan Q untuk keluar.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca frame.")
        break

    # Mirror
    frame = cv2.flip(frame, 1)

    # BGR → RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    left_state = "N/A"
    right_state = "N/A"

    left_prob = 0.0
    right_prob = 0.0

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # ----------------------------------------------------
        # LEFT EYE
        # ----------------------------------------------------

        left_crop, left_box = get_eye_crop(
            frame,
            landmarks,
            LEFT_EYE
        )

        left_state, left_prob = predict_eye(left_crop)

        # ----------------------------------------------------
        # RIGHT EYE
        # ----------------------------------------------------

        right_crop, right_box = get_eye_crop(
            frame,
            landmarks,
            RIGHT_EYE
        )

        right_state, right_prob = predict_eye(right_crop)

        # ----------------------------------------------------
        # DRAW EYE BOXES
        # ----------------------------------------------------

        lx1, ly1, lx2, ly2 = left_box
        rx1, ry1, rx2, ry2 = right_box

        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            (255, 0, 0),
            2
        )

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (255, 0, 0),
            2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"Left Eye : {left_state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Right Eye: {right_state}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "CNN Eye State Detection",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
face_mesh.close()
cv2.destroyAllWindows()