import cv2
import mediapipe as mp
import math
import time


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
# MediaPipe
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

# Mata harus tertutup selama ini
# sebelum dianggap long closure.
LONG_CLOSURE_THRESHOLD = 1.0


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()


print("Drowsiness eye closure test")
print("Tekan Q untuk keluar.")


# =========================
# State
# =========================

eyes_closed = False
closure_start_time = None
closure_duration = 0.0


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

        # Ambil landmark mata
        left_eye_points = [
            face_landmarks.landmark[index]
            for index in LEFT_EYE
        ]

        right_eye_points = [
            face_landmarks.landmark[index]
            for index in RIGHT_EYE
        ]

        # Hitung EAR masing-masing mata
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        # Rata-rata
        ear = (left_ear + right_ear) / 2.0


        # =========================
        # Tentukan mata tertutup
        # =========================

        if ear < EAR_THRESHOLD:

            # Baru mulai menutup
            if not eyes_closed:

                eyes_closed = True
                closure_start_time = time.time()

            # Hitung durasi
            closure_duration = time.time() - closure_start_time

        else:

            # Mata terbuka kembali
            eyes_closed = False
            closure_start_time = None
            closure_duration = 0.0


        # =========================
        # Tentukan status
        # =========================

        if eyes_closed:

            if closure_duration >= LONG_CLOSURE_THRESHOLD:

                status = "LONG EYE CLOSURE"

            else:

                status = "EYES CLOSED"

        else:

            status = "EYES OPEN"


        # =========================
        # Warna status
        # =========================

        if status == "LONG EYE CLOSURE":

            text_color = (0, 0, 255)

        elif status == "EYES CLOSED":

            text_color = (0, 165, 255)

        else:

            text_color = (0, 255, 0)


        # =========================
        # Tampilkan informasi
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

        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"Closed: {closure_duration:.2f}s",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )


        # =========================
        # Tampilkan landmark mata
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
    # Show camera
    # =========================

    cv2.imshow(
        "Drowsiness - Eye Closure",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
face_mesh.close()
cv2.destroyAllWindows()