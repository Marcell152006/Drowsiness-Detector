import cv2
import mediapipe as mp
import math


# =========================
# FUNCTION: Hitung jarak 2 titik
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
    """
    eye_landmarks berisi 6 titik:

    p1 -------- p4
      p2      p3
      p6      p5
    """

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# =========================
# MediaPipe Face Mesh
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


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

# Mata kiri
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

# Mata kanan
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("ERROR: Webcam tidak dapat dibuka.")
    exit()


print("Webcam berhasil dibuka.")
print("Tekan Q untuk keluar.")


# =========================
# Main Loop
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak bisa membaca frame.")
        break

    # Mirror seperti kamera laptop
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        # Karena max_num_faces = 1
        face_landmarks = results.multi_face_landmarks[0]

        # Ambil landmark mata kiri
        left_eye_points = [
            face_landmarks.landmark[index]
            for index in LEFT_EYE
        ]

        # Ambil landmark mata kanan
        right_eye_points = [
            face_landmarks.landmark[index]
            for index in RIGHT_EYE
        ]

        # Hitung EAR
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        # Rata-rata kedua mata
        ear = (left_ear + right_ear) / 2.0


        # =========================
        # Tampilkan titik mata
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


        # =========================
        # Status mata
        # =========================

        if ear < 0.21:
            eye_status = "CLOSED"
        else:
            eye_status = "OPEN"


        # =========================
        # Tampilkan informasi
        # =========================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No Face Detected",
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
        "Drowsiness Detection - EAR",
        frame
    )


    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Cleanup
# =========================

cap.release()
face_mesh.close()
cv2.destroyAllWindows()