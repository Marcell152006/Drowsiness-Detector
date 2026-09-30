import cv2
import mediapipe as mp
import math
import time
from collections import deque


# =========================================================
# FUNCTION: Euclidean Distance
# =========================================================
def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# =========================================================
# FUNCTION: Calculate EAR
# =========================================================
def calculate_ear(eye_landmarks):

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# =========================================================
# MEDIAPIPE
# =========================================================
mp_face_mesh = mp.solutions.face_mesh

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
# PARAMETERS
# =========================================================

# EAR di bawah nilai ini dianggap CLOSED
EAR_THRESHOLD = 0.21

# Mata tertutup selama ini → DROWSY langsung
LONG_CLOSURE_THRESHOLD = 1.2

# Window PERCLOS dibuat lebih pendek
PERCLOS_WINDOW = 5.0

# Threshold PERCLOS
PERCLOS_WARNING = 20.0
PERCLOS_DROWSY = 40.0


# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()

print("===================================")
print("     DROWSINESS DETECTION TEST")
print("===================================")
print("Tekan Q untuk keluar.")


# =========================================================
# STATE
# =========================================================

eyes_closed = False

closure_start_time = None

closure_duration = 0.0

# Menyimpan:
# (timestamp, is_closed)
eye_history = deque()

# Status terakhir
current_status = "NORMAL"


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak bisa membaca frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # -------------------------------------------------
        # Get eye landmarks
        # -------------------------------------------------

        left_eye_points = [
            face_landmarks.landmark[index]
            for index in LEFT_EYE
        ]

        right_eye_points = [
            face_landmarks.landmark[index]
            for index in RIGHT_EYE
        ]


        # -------------------------------------------------
        # Calculate EAR
        # -------------------------------------------------

        left_ear = calculate_ear(left_eye_points)

        right_ear = calculate_ear(right_eye_points)

        ear = (left_ear + right_ear) / 2.0


        # =================================================
        # EYE STATE
        # =================================================

        is_closed = ear < EAR_THRESHOLD

        current_time = time.time()


        # =================================================
        # CLOSURE TIMER
        # =================================================

        if is_closed:

            if not eyes_closed:

                eyes_closed = True

                closure_start_time = current_time

            closure_duration = (
                current_time - closure_start_time
            )

        else:

            eyes_closed = False

            closure_start_time = None

            closure_duration = 0.0


        # =================================================
        # PERCLOS HISTORY
        # =================================================

        eye_history.append(
            (current_time, is_closed)
        )


        # Hapus data yang lebih lama
        # dari window
        while (
            eye_history
            and current_time - eye_history[0][0]
            > PERCLOS_WINDOW
        ):
            eye_history.popleft()


        # =================================================
        # TIME-WEIGHTED PERCLOS
        # =================================================

        closed_duration = 0.0

        total_duration = 0.0

        if len(eye_history) >= 2:

            history = list(eye_history)

            for i in range(1, len(history)):

                previous_time, previous_closed = history[i - 1]

                current_time_item, _ = history[i]

                dt = current_time_item - previous_time

                # Batasi interval agar tidak terlalu besar
                dt = min(dt, 0.2)

                total_duration += dt

                if previous_closed:

                    closed_duration += dt


        if total_duration > 0:

            perclos = (
                closed_duration /
                total_duration
            ) * 100.0

        else:

            perclos = 0.0


        # =================================================
        # DROWSINESS DECISION
        # =================================================

        # PRIORITY 1:
        # Long eye closure
        if closure_duration >= LONG_CLOSURE_THRESHOLD:

            current_status = "DROWSY"

            reason = "LONG EYE CLOSURE"


        # PRIORITY 2:
        # High PERCLOS
        elif perclos >= PERCLOS_DROWSY:

            current_status = "DROWSY"

            reason = "HIGH PERCLOS"


        # PRIORITY 3:
        # Warning
        elif perclos >= PERCLOS_WARNING:

            current_status = "WARNING"

            reason = "ELEVATED PERCLOS"


        # PRIORITY 4:
        # Normal
        else:

            current_status = "NORMAL"

            reason = "ALERT"


        # =================================================
        # STATUS COLOR
        # =================================================

        if current_status == "DROWSY":

            color = (0, 0, 255)

        elif current_status == "WARNING":

            color = (0, 165, 255)

        else:

            color = (0, 255, 0)


        # =================================================
        # DISPLAY
        # =================================================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        eye_state = (
            "CLOSED"
            if is_closed
            else "OPEN"
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_state}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        cv2.putText(
            frame,
            f"Closed: {closure_duration:.2f}s",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        cv2.putText(
            frame,
            f"PERCLOS: {perclos:.1f}%",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        cv2.putText(
            frame,
            f"STATUS: {current_status}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )


        cv2.putText(
            frame,
            f"Reason: {reason}",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        # =================================================
        # DRAW EYE LANDMARKS
        # =================================================

        h, w, _ = frame.shape

        for point in (
            left_eye_points +
            right_eye_points
        ):

            x = int(point.x * w)

            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


    # =====================================================
    # NO FACE
    # =====================================================

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "Drowsiness Detection",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()