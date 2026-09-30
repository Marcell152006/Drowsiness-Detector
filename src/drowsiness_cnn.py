import cv2
import time
import threading
import winsound

import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/eye_state_cnn.keras"

IMG_SIZE = 64

# CNN threshold
CNN_THRESHOLD = 0.5

# Long eye closure
LONG_CLOSURE_THRESHOLD = 1.2  # seconds

# PERCLOS
PERCLOS_WINDOW = 5.0          # seconds
PERCLOS_WARNING = 20.0        # %
PERCLOS_DROWSY = 40.0         # %

# Camera
CAMERA_ID = 4

# Alarm cooldown
ALARM_COOLDOWN = 3.0          # seconds


# =========================================================
# LOAD CNN MODEL
# =========================================================

print("Loading CNN model...")

model = keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")


# =========================================================
# MEDIAPIPE FACE MESH
# =========================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# =========================================================
# GET EYE CROP
# =========================================================

def get_eye_crop(frame, landmarks, eye_indices):

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

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    # Padding
    padding_x = int(eye_width * 0.5)
    padding_y = int(eye_height * 1.0)

    x1 = max(0, x_min - padding_x)
    x2 = min(w, x_max + padding_x)

    y1 = max(0, y_min - padding_y)
    y2 = min(h, y_max + padding_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


# =========================================================
# CNN PREDICTION
# =========================================================

def predict_eye(crop):

    if crop is None:
        return None, 0.0

    # BGR → RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(
        rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(
        normalized,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        input_image,
        verbose=0
    )[0][0]

    # Dataset:
    # Closed_Eyes = 0
    # Open_Eyes   = 1

    if prediction >= CNN_THRESHOLD:

        state = "OPEN"
        confidence = prediction

    else:

        state = "CLOSED"
        confidence = 1.0 - prediction

    return state, confidence


# =========================================================
# PERCLOS CALCULATION
# =========================================================

def calculate_perclos(history):

    if len(history) < 2:
        return 0.0

    total_duration = 0.0
    closed_duration = 0.0

    for i in range(1, len(history)):

        previous_time, previous_closed = history[i - 1]
        current_time, _ = history[i]

        dt = current_time - previous_time

        # Prevent abnormal time gaps
        dt = min(dt, 0.2)

        total_duration += dt

        if previous_closed:
            closed_duration += dt

    if total_duration <= 0:
        return 0.0

    perclos = (
        closed_duration /
        total_duration
    ) * 100.0

    return perclos


# =========================================================
# DETERMINE DROWSINESS STATUS
# =========================================================

def determine_status(
    long_closure_duration,
    perclos
):

    # Priority 1:
    # Long eye closure
    if long_closure_duration >= LONG_CLOSURE_THRESHOLD:

        return "DROWSY"

    # Priority 2:
    # High PERCLOS
    if perclos >= PERCLOS_DROWSY:

        return "DROWSY"

    # Priority 3:
    # Warning
    if perclos >= PERCLOS_WARNING:

        return "WARNING"

    return "NORMAL"


# =========================================================
# AUDIO FUNCTIONS
# =========================================================

def play_warning():

    winsound.Beep(
        800,
        250
    )


def play_alarm():

    winsound.Beep(
        1000,
        400
    )

    winsound.Beep(
        1200,
        400
    )

    winsound.Beep(
        1000,
        400
    )


def run_alarm(alarm_type):

    """
    Run alarm inside a background thread
    so the camera loop does not freeze.
    """

    if alarm_type == "WARNING":

        play_warning()

    elif alarm_type == "DROWSY":

        play_alarm()


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Cannot open camera.")

    exit()


# Optional camera resolution
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# =========================================================
# STATE VARIABLES
# =========================================================

closure_start_time = None

history = deque()

last_alarm_time = 0.0

previous_time = time.time()

fps = 0.0


# =========================================================
# MAIN LOOP
# =========================================================

print()
print("======================================")
print(" DROWSINESS DETECTION STARTED")
print("======================================")
print()
print("Press Q to quit.")
print()


while True:

    # -----------------------------------------------------
    # READ CAMERA
    # -----------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Cannot read camera.")

        break


    # Mirror image
    frame = cv2.flip(
        frame,
        1
    )


    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    current_time = time.time()

    frame_dt = current_time - previous_time

    previous_time = current_time

    if frame_dt > 0:

        current_fps = 1.0 / frame_dt

        # Smooth FPS
        fps = (
            fps * 0.9 +
            current_fps * 0.1
        )


    # -----------------------------------------------------
    # MEDIAPIPE
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb_frame
    )


    status = "NO FACE"

    perclos = 0.0

    long_closure_duration = 0.0

    left_state = "UNKNOWN"
    right_state = "UNKNOWN"

    left_confidence = 0.0
    right_confidence = 0.0


    # -----------------------------------------------------
    # FACE DETECTED
    # -----------------------------------------------------

    if results.multi_face_landmarks:

        face_landmarks = (
            results.multi_face_landmarks[0]
        )

        landmarks = face_landmarks.landmark


        # -------------------------------------------------
        # LEFT EYE
        # -------------------------------------------------

        left_crop, left_box = get_eye_crop(
            frame,
            landmarks,
            LEFT_EYE
        )

        if left_crop is not None:

            left_state, left_confidence = predict_eye(
                left_crop
            )


        # -------------------------------------------------
        # RIGHT EYE
        # -------------------------------------------------

        right_crop, right_box = get_eye_crop(
            frame,
            landmarks,
            RIGHT_EYE
        )

        if right_crop is not None:

            right_state, right_confidence = predict_eye(
                right_crop
            )


        # -------------------------------------------------
        # DRAW EYE BOXES
        # -------------------------------------------------

        if left_box is not None:

            x1, y1, x2, y2 = left_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


        if right_box is not None:

            x1, y1, x2, y2 = right_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


        # -------------------------------------------------
        # BOTH EYES CLOSED
        # -------------------------------------------------

        both_eyes_closed = (
            left_state == "CLOSED"
            and
            right_state == "CLOSED"
        )


        # -------------------------------------------------
        # LONG EYE CLOSURE
        # -------------------------------------------------

        if both_eyes_closed:

            if closure_start_time is None:

                closure_start_time = time.time()

            long_closure_duration = (
                time.time()
                - closure_start_time
            )

        else:

            closure_start_time = None

            long_closure_duration = 0.0


        # -------------------------------------------------
        # PERCLOS HISTORY
        # -------------------------------------------------

        now = time.time()

        history.append(
            (
                now,
                both_eyes_closed
            )
        )


        # Remove samples outside window

        while (
            history
            and
            now - history[0][0] > PERCLOS_WINDOW
        ):

            history.popleft()


        # Calculate PERCLOS

        perclos = calculate_perclos(
            history
        )


        # -------------------------------------------------
        # DETERMINE STATUS
        # -------------------------------------------------

        status = determine_status(
            long_closure_duration,
            perclos
        )


        # -------------------------------------------------
        # AUDIO ALERT
        # -------------------------------------------------

        if status == "DROWSY":

            if (
                now - last_alarm_time
                >= ALARM_COOLDOWN
            ):

                threading.Thread(
                    target=run_alarm,
                    args=("DROWSY",),
                    daemon=True
                ).start()

                last_alarm_time = now


        elif status == "WARNING":

            if (
                now - last_alarm_time
                >= ALARM_COOLDOWN
            ):

                threading.Thread(
                    target=run_alarm,
                    args=("WARNING",),
                    daemon=True
                ).start()

                last_alarm_time = now


    # =====================================================
    # UI
    # =====================================================

    # Status
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0)
        if status == "NORMAL"
        else
        (0, 255, 255)
        if status == "WARNING"
        else
        (0, 0, 255),
        2
    )


    # Left eye
    cv2.putText(
        frame,
        f"LEFT: {left_state} ({left_confidence:.2f})",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # Right eye
    cv2.putText(
        frame,
        f"RIGHT: {right_state} ({right_confidence:.2f})",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # PERCLOS
    cv2.putText(
        frame,
        f"PERCLOS: {perclos:.1f}%",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Long closure
    cv2.putText(
        frame,
        f"CLOSURE: {long_closure_duration:.1f}s",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "Drowsiness Detection - CNN",
        frame
    )


    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

print()
print("Drowsiness detection stopped.")