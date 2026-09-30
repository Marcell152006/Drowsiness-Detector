import cv2
import mediapipe as mp
import math
import time
from collections import deque


# =========================
# FUNCTION: Jarak 2 titik
# =========================
def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# =========================
# FUNCTION: Hitung EAR
# =========================
def calculate_ear(eye_landmarks):

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# =========================
# MediaPipe Face Mesh
# =========================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# Landmark mata
# =========================
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


# =========================
# Parameter
# =========================

EAR_THRESHOLD = 0.21

# Window PERCLOS
WINDOW_SECONDS = 10.0


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()


print("PERCLOS Test")
print("Window:", WINDOW_SECONDS, "seconds")
print("Tekan Q untuk keluar.")


# =========================
# Simpan data eye state
# =========================

# Setiap data:
# (timestamp, is_closed)

eye_history = deque()


# =========================
# Main Loop
# =========================
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


    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]


        # =========================
        # Ambil landmark mata
        # =========================

        left_eye_points = [
            face_landmarks.landmark[index]
            for index in LEFT_EYE
        ]

        right_eye_points = [
            face_landmarks.landmark[index]
            for index in RIGHT_EYE
        ]


        # =========================
        # Hitung EAR
        # =========================

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        ear = (left_ear + right_ear) / 2.0


        # =========================
        # Eye state
        # =========================

        is_closed = ear < EAR_THRESHOLD


        # =========================
        # Simpan history
        # =========================

        current_time = time.time()

        eye_history.append(
            (current_time, is_closed)
        )


        # =========================
        # Hapus data
        # yang lebih lama
        # dari window
        # =========================

        while (
            eye_history
            and current_time - eye_history[0][0] > WINDOW_SECONDS
        ):
            eye_history.popleft()


        # =========================
        # Hitung PERCLOS
        # =========================

        if len(eye_history) > 1:

            closed_frames = sum(
                1
                for _, closed in eye_history
                if closed
            )

            total_frames = len(eye_history)

            perclos = (
                closed_frames / total_frames
            ) * 100.0

        else:

            perclos = 0.0


        # =========================
        # Status
        # =========================

        if perclos >= 40:

            status = "DROWSY"

        elif perclos >= 25:

            status = "WARNING"

        else:

            status = "NORMAL"


        # =========================
        # Warna status
        # =========================

        if status == "DROWSY":

            color = (0, 0, 255)

        elif status == "WARNING":

            color = (0, 165, 255)

        else:

            color = (0, 255, 0)


        # =========================
        # Tampilkan EAR
        # =========================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # =========================
        # Tampilkan Eye State
        # =========================

        eye_state = "CLOSED" if is_closed else "OPEN"

        cv2.putText(
            frame,
            f"Eyes: {eye_state}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        # =========================
        # Tampilkan PERCLOS
        # =========================

        cv2.putText(
            frame,
            f"PERCLOS: {perclos:.1f}%",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        # =========================
        # Tampilkan Status
        # =========================

        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        # =========================
        # Landmark mata
        # =========================

        h, w, _ = frame.shape

        for point in left_eye_points + right_eye_points:

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


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


    # =========================
    # Tampilkan kamera
    # =========================

    cv2.imshow(
        "Drowsiness - PERCLOS",
        frame
    )


    # =========================
    # Q untuk keluar
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
face_mesh.close()
cv2.destroyAllWindows()