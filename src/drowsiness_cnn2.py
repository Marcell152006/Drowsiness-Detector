import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import winsound
import threading
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/eye_state_cnn.keras"

IMG_SIZE = 64
CNN_THRESHOLD = 0.5


# ============================================================
# EYE CLOSURE
# ============================================================

# Dibuat lebih tidak sensitif.
# Mata harus tertutup sekitar 1 detik untuk DROWSY.
LONG_CLOSURE_THRESHOLD = 1.0


# ============================================================
# PERCLOS
# ============================================================

PERCLOS_WINDOW = 5.0

# Jangan terlalu cepat WARNING
PERCLOS_WARNING = 35.0

# DROWSY hanya jika PERCLOS sangat tinggi
PERCLOS_DROWSY = 50.0


# ============================================================
# CAMERA
# ============================================================

CAMERA_ID = 4


# ============================================================
# HORIZONTAL HEAD
# ============================================================

MAX_HORIZONTAL_DEVIATION = 0.22


# ============================================================
# VERTICAL HEAD
# ============================================================

# Threshold dibuat lebih konservatif.
#
# Nilai ini bukan langsung menentukan status.
# Kita juga memakai beberapa frame stabil.

UP_THRESHOLD = -0.35
DOWN_THRESHOLD = 0.35


# ============================================================
# HEAD DIRECTION STABILITY
# ============================================================

# Berapa banyak frame berturut-turut harus sama
# sebelum dianggap benar-benar UP/DOWN.

HEAD_DIRECTION_CONFIRM_FRAMES = 8


# ============================================================
# FRONT STABILIZATION
# ============================================================

FRONT_STABLE_TIME = 0.7


# ============================================================
# LOOKING DOWN
# ============================================================

DOWN_WARNING_TIME = 5.0
DOWN_DROWSY_TIME = 10.0


# ============================================================
# DROWSY RECOVERY
# ============================================================

EYES_OPEN_RECOVERY_TIME = 0.25


# ============================================================
# ALARM
# ============================================================

ALARM_COOLDOWN = 4.0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")


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
# EYE LANDMARKS
# ============================================================

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


# ============================================================
# GET EYE CROP
# ============================================================

def get_eye_crop(
    frame,
    landmarks,
    eye_indices
):

    h, w = frame.shape[:2]

    points = []

    for idx in eye_indices:

        x = int(
            landmarks[idx].x * w
        )

        y = int(
            landmarks[idx].y * h
        )

        points.append(
            (x, y)
        )

    points = np.array(points)

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    if eye_width <= 0 or eye_height <= 0:
        return None, None

    padding_x = int(
        eye_width * 0.5
    )

    padding_y = int(
        eye_height * 1.0
    )

    x1 = max(
        0,
        x_min - padding_x
    )

    x2 = min(
        w,
        x_max + padding_x
    )

    y1 = max(
        0,
        y_min - padding_y
    )

    y2 = min(
        h,
        y_max + padding_y
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None, None

    return crop, (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# PREPARE EYE
# ============================================================

def prepare_eye(crop):

    if crop is None:
        return None

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop_resized = cv2.resize(
        crop_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    crop_normalized = (
        crop_resized.astype(
            np.float32
        ) / 255.0
    )

    return crop_normalized


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_eyes(
    left_crop,
    right_crop
):

    left_input = prepare_eye(
        left_crop
    )

    right_input = prepare_eye(
        right_crop
    )

    if (
        left_input is None
        or
        right_input is None
    ):
        return None, None

    batch = np.array([
        left_input,
        right_input
    ])

    predictions = model.predict(
        batch,
        verbose=0
    ).flatten()

    left_prediction = float(
        predictions[0]
    )

    right_prediction = float(
        predictions[1]
    )

    if left_prediction >= CNN_THRESHOLD:

        left_state = "OPEN"
        left_confidence = left_prediction

    else:

        left_state = "CLOSED"
        left_confidence = 1.0 - left_prediction

    if right_prediction >= CNN_THRESHOLD:

        right_state = "OPEN"
        right_confidence = right_prediction

    else:

        right_state = "CLOSED"
        right_confidence = 1.0 - right_prediction

    return (
        (
            left_state,
            left_confidence
        ),
        (
            right_state,
            right_confidence
        )
    )


# ============================================================
# HEAD ORIENTATION
# ============================================================

def get_head_orientation(
    landmarks
):

    # --------------------------------------------------------
    # Horizontal face landmarks
    # --------------------------------------------------------

    left_face = landmarks[234]
    right_face = landmarks[454]

    nose = landmarks[1]


    # --------------------------------------------------------
    # Eye center
    # --------------------------------------------------------

    left_eye_center_y = (
        landmarks[159].y +
        landmarks[145].y
    ) / 2.0

    right_eye_center_y = (
        landmarks[386].y +
        landmarks[374].y
    ) / 2.0

    eye_line_y = (
        left_eye_center_y +
        right_eye_center_y
    ) / 2.0


    # --------------------------------------------------------
    # Mouth center
    # --------------------------------------------------------

    mouth_center_y = (
        landmarks[13].y +
        landmarks[14].y
    ) / 2.0


    # ========================================================
    # HORIZONTAL
    # ========================================================

    face_center_x = (
        left_face.x +
        right_face.x
    ) / 2.0

    face_width = abs(
        right_face.x -
        left_face.x
    )

    if face_width < 0.001:
        face_width = 0.001

    horizontal_deviation = (
        nose.x -
        face_center_x
    ) / face_width


    # ========================================================
    # VERTICAL
    # ========================================================

    eye_to_mouth = (
        mouth_center_y -
        eye_line_y
    )

    if eye_to_mouth < 0.001:
        eye_to_mouth = 0.001


    # Posisi hidung relatif terhadap
    # garis mata -> mulut.
    #
    # FRONT kira-kira berada di tengah.
    #
    # Jika nose bergerak jauh ke atas:
    # kepala melihat UP.
    #
    # Jika nose bergerak jauh ke bawah:
    # kepala melihat DOWN.

    vertical_position = (
        nose.y -
        eye_line_y
    ) / eye_to_mouth


    # Normalisasi terhadap posisi tengah.
    #
    # Nilai sekitar 0.5 adalah posisi normal.
    #
    # Semakin kecil -> UP
    # Semakin besar -> DOWN

    vertical_deviation = (
        vertical_position -
        0.50
    )


    # ========================================================
    # HORIZONTAL STATE
    # ========================================================

    if (
        horizontal_deviation
        < -MAX_HORIZONTAL_DEVIATION
    ):

        horizontal_state = "LEFT"

    elif (
        horizontal_deviation
        > MAX_HORIZONTAL_DEVIATION
    ):

        horizontal_state = "RIGHT"

    else:

        horizontal_state = "FRONT"


    # ========================================================
    # RAW VERTICAL STATE
    # ========================================================

    if (
        vertical_deviation
        < UP_THRESHOLD
    ):

        vertical_state = "UP"

    elif (
        vertical_deviation
        > DOWN_THRESHOLD
    ):

        vertical_state = "DOWN"

    else:

        vertical_state = "FRONT"


    return (
        horizontal_deviation,
        vertical_deviation,
        horizontal_state,
        vertical_state
    )


# ============================================================
# PERCLOS
# ============================================================

def calculate_perclos(
    history
):

    if len(history) < 2:
        return 0.0

    now = time.time()

    recent_history = [
        item
        for item in history
        if now - item[0]
        <= PERCLOS_WINDOW
    ]

    if len(recent_history) < 2:
        return 0.0

    total_duration = 0.0
    closed_duration = 0.0

    for i in range(
        1,
        len(recent_history)
    ):

        previous_time, previous_closed = (
            recent_history[i - 1]
        )

        current_time, _ = (
            recent_history[i]
        )

        dt = (
            current_time -
            previous_time
        )

        dt = min(
            dt,
            0.2
        )

        total_duration += dt

        if previous_closed:
            closed_duration += dt

    if total_duration <= 0:
        return 0.0

    return (
        closed_duration /
        total_duration
    ) * 100.0


# ============================================================
# ALARM
# ============================================================

def play_warning():

    try:

        winsound.Beep(
            800,
            250
        )

    except Exception:
        pass


def play_drowsy():

    try:

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

    except Exception:
        pass


def trigger_alarm(
    status
):

    if status == "WARNING":

        threading.Thread(
            target=play_warning,
            daemon=True
        ).start()

    elif status == "DROWSY":

        threading.Thread(
            target=play_drowsy,
            daemon=True
        ).start()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not cap.isOpened():

    print(
        f"ERROR: Camera {CAMERA_ID} could not be opened."
    )

    raise SystemExit


print()
print("Drowsiness detection started.")
print("Press Q to quit.")
print()


# ============================================================
# STATE VARIABLES
# ============================================================

history = deque()

closure_start_time = None

down_start_time = None

front_start_time = None

eyes_open_start_time = None

previous_status = "NORMAL"

current_status = "NORMAL"

last_alarm_time = 0.0


# ============================================================
# HEAD STABILITY
# ============================================================

candidate_head_state = "FRONT"
candidate_head_frames = 0

stable_head_state = "FRONT"


# ============================================================
# FPS
# ============================================================

fps = 0.0
fps_counter = 0
fps_start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.flip(
        frame,
        1
    )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = face_mesh.process(
        rgb
    )


    # ========================================================
    # DEFAULT
    # ========================================================

    head_text = "NO FACE"

    horizontal_deviation = 0.0

    vertical_deviation = 0.0

    left_result = None
    right_result = None

    perclos = 0.0

    closure_duration = 0.0

    down_duration = 0.0


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if results.multi_face_landmarks:

        face_landmarks = (
            results.multi_face_landmarks[0]
        )

        landmarks = (
            face_landmarks.landmark
        )


        # ----------------------------------------------------
        # Head orientation
        # ----------------------------------------------------

        (
            horizontal_deviation,
            vertical_deviation,
            horizontal_state,
            vertical_state
        ) = get_head_orientation(
            landmarks
        )


        # ----------------------------------------------------
        # Draw contours
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None
        )


        # ====================================================
        # HORIZONTAL
        # ====================================================

        if horizontal_state != "FRONT":

            # Sideways langsung dianggap valid
            # karena horizontal detection sebelumnya
            # sudah cukup stabil.

            stable_head_state = (
                horizontal_state
            )

            candidate_head_frames = 0

            if horizontal_state == "LEFT":

                head_text = "HEAD LEFT"

            else:

                head_text = "HEAD RIGHT"


            current_status = "HEAD TURNED"


            history.clear()

            closure_start_time = None
            eyes_open_start_time = None
            down_start_time = None
            front_start_time = None


        else:

            # =================================================
            # VERTICAL STABILITY
            # =================================================

            if vertical_state == candidate_head_state:

                candidate_head_frames += 1

            else:

                candidate_head_state = (
                    vertical_state
                )

                candidate_head_frames = 1


            # Hanya ubah stable state jika
            # sudah cukup banyak frame konsisten.

            if (
                candidate_head_frames
                >= HEAD_DIRECTION_CONFIRM_FRAMES
            ):

                stable_head_state = (
                    candidate_head_state
                )


            # =================================================
            # HEAD UP
            # =================================================

            if stable_head_state == "UP":

                head_text = "HEAD UP"

                current_status = "HEAD UP"


                # Tidak ada CNN
                # Tidak ada PERCLOS
                # Tidak ada alarm

                history.clear()

                closure_start_time = None
                eyes_open_start_time = None
                down_start_time = None
                front_start_time = None


            # =================================================
            # HEAD DOWN
            # =================================================

            elif stable_head_state == "DOWN":

                head_text = "HEAD DOWN"


                if down_start_time is None:

                    down_start_time = (
                        time.time()
                    )


                down_duration = (
                    time.time()
                    -
                    down_start_time
                )


                # --------------------------------------------
                # < 5 SEC
                # --------------------------------------------

                if (
                    down_duration
                    < DOWN_WARNING_TIME
                ):

                    current_status = "NORMAL"


                # --------------------------------------------
                # 5 - 10 SEC
                # --------------------------------------------

                elif (
                    down_duration
                    < DOWN_DROWSY_TIME
                ):

                    current_status = "WARNING"


                # --------------------------------------------
                # >= 10 SEC
                # --------------------------------------------

                else:

                    current_status = "DROWSY"


                # Eye detection disabled while DOWN.

                history.clear()

                closure_start_time = None
                eyes_open_start_time = None
                front_start_time = None


            # =================================================
            # FRONT
            # =================================================

            else:

                head_text = "FRONT"

                down_start_time = None


                # ------------------------------------------------
                # Front stabilization
                # ------------------------------------------------

                if front_start_time is None:

                    front_start_time = (
                        time.time()
                    )


                front_duration = (
                    time.time()
                    -
                    front_start_time
                )


                if (
                    front_duration
                    < FRONT_STABLE_TIME
                ):

                    current_status = (
                        "STABILIZING"
                    )

                    history.clear()

                    closure_start_time = None
                    eyes_open_start_time = None


                else:

                    # =================================================
                    # EYE CROPS
                    # =================================================

                    left_crop, left_box = (
                        get_eye_crop(
                            frame,
                            landmarks,
                            LEFT_EYE
                        )
                    )

                    right_crop, right_box = (
                        get_eye_crop(
                            frame,
                            landmarks,
                            RIGHT_EYE
                        )
                    )


                    # =================================================
                    # CNN
                    # =================================================

                    (
                        left_result,
                        right_result
                    ) = predict_eyes(
                        left_crop,
                        right_crop
                    )


                    if (
                        left_result is not None
                        and
                        right_result is not None
                    ):

                        (
                            left_state,
                            left_confidence
                        ) = left_result

                        (
                            right_state,
                            right_confidence
                        ) = right_result


                        # ---------------------------------------------
                        # Draw eye boxes
                        # ---------------------------------------------

                        if left_box is not None:

                            x1, y1, x2, y2 = (
                                left_box
                            )

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                2
                            )


                        if right_box is not None:

                            x1, y1, x2, y2 = (
                                right_box
                            )

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                2
                            )


                        # =================================================
                        # EYE STATE
                        # =================================================

                        both_closed = (
                            left_state == "CLOSED"
                            and
                            right_state == "CLOSED"
                        )


                        both_open = (
                            left_state == "OPEN"
                            and
                            right_state == "OPEN"
                        )


                        now = time.time()


                        # =================================================
                        # PERCLOS HISTORY
                        # =================================================

                        history.append(
                            (
                                now,
                                both_closed
                            )
                        )


                        while (
                            history
                            and
                            now - history[0][0]
                            > PERCLOS_WINDOW
                        ):

                            history.popleft()


                        perclos = (
                            calculate_perclos(
                                history
                            )
                        )


                        # =================================================
                        # EYE CLOSURE TIMER
                        # =================================================

                        if both_closed:

                            eyes_open_start_time = None


                            if (
                                closure_start_time
                                is None
                            ):

                                closure_start_time = now


                            closure_duration = (
                                now
                                -
                                closure_start_time
                            )


                        else:

                            closure_duration = 0.0


                            # =========================================
                            # OPEN EYES
                            # =========================================

                            if both_open:

                                if (
                                    eyes_open_start_time
                                    is None
                                ):

                                    eyes_open_start_time = now


                                eyes_open_duration = (
                                    now
                                    -
                                    eyes_open_start_time
                                )


                                # -------------------------------------
                                # DROWSY RECOVERY
                                # -------------------------------------

                                if (
                                    previous_status
                                    == "DROWSY"
                                    and
                                    eyes_open_duration
                                    >= EYES_OPEN_RECOVERY_TIME
                                ):

                                    current_status = (
                                        "NORMAL"
                                    )


                                    closure_start_time = (
                                        None
                                    )


                                    history.clear()

                                    perclos = 0.0


                            else:

                                eyes_open_start_time = (
                                    None
                                )


                            closure_start_time = (
                                None
                            )


                        # =================================================
                        # STATUS DECISION
                        # =================================================

                        # If already recovered from DROWSY,
                        # don't immediately re-trigger based
                        # on old PERCLOS.

                        if not (
                            previous_status == "DROWSY"
                            and
                            current_status == "NORMAL"
                        ):

                            # ---------------------------------------------
                            # LONG CLOSURE
                            # ---------------------------------------------

                            if (
                                closure_duration
                                >= LONG_CLOSURE_THRESHOLD
                            ):

                                current_status = (
                                    "DROWSY"
                                )


                            # ---------------------------------------------
                            # PERCLOS DROWSY
                            # ---------------------------------------------

                            elif (
                                perclos
                                >= PERCLOS_DROWSY
                            ):

                                current_status = (
                                    "DROWSY"
                                )


                            # ---------------------------------------------
                            # PERCLOS WARNING
                            # ---------------------------------------------

                            elif (
                                perclos
                                >= PERCLOS_WARNING
                            ):

                                current_status = (
                                    "WARNING"
                                )


                            else:

                                current_status = (
                                    "NORMAL"
                                )


    # ========================================================
    # NO FACE
    # ========================================================

    else:

        current_status = "NO FACE"

        head_text = "NO FACE"

        history.clear()

        closure_start_time = None
        eyes_open_start_time = None
        down_start_time = None
        front_start_time = None

        candidate_head_state = "FRONT"
        candidate_head_frames = 0
        stable_head_state = "FRONT"


    # ========================================================
    # ALARM
    # ========================================================

    now = time.time()


    if current_status in (
        "WARNING",
        "DROWSY"
    ):

        if (
            now - last_alarm_time
            >= ALARM_COOLDOWN
        ):

            trigger_alarm(
                current_status
            )

            last_alarm_time = now


    # ========================================================
    # SAVE PREVIOUS STATUS
    # ========================================================

    previous_status = (
        current_status
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1

    elapsed = (
        time.time()
        -
        fps_start_time
    )


    if elapsed >= 1.0:

        fps = (
            fps_counter /
            elapsed
        )

        fps_counter = 0

        fps_start_time = (
            time.time()
        )


    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        frame,
        f"STATUS: {current_status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"HEAD: {head_text}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"H: {horizontal_deviation:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"V: {vertical_deviation:.2f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    if (
        left_result is not None
        and
        right_result is not None
    ):

        (
            left_state,
            left_confidence
        ) = left_result

        (
            right_state,
            right_confidence
        ) = right_result


        cv2.putText(
            frame,
            f"LEFT: {left_state} "
            f"{left_confidence * 100:.1f}%",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"RIGHT: {right_state} "
            f"{right_confidence * 100:.1f}%",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    cv2.putText(
        frame,
        f"PERCLOS: {perclos:.1f}%",
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"CLOSURE: {closure_duration:.2f}s",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"DOWN TIME: {down_duration:.1f}s",
        (20, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Drowsiness Detection - CNN",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

print(
    "Drowsiness detection stopped."
)