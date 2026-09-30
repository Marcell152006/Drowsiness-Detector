import cv2

# =========================
# LOAD CASCADES
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty():
    print("Gagal memuat face cascade.")
    exit()

if eye_cascade.empty():
    print("Gagal memuat eye cascade.")
    exit()


# =========================
# OPEN WEBCAM
# =========================

cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()

print("Webcam aktif.")
print("Tekan Q untuk keluar.")


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame.")
        break

    # Mirror
    frame = cv2.flip(frame, 1)

    # =========================
    # GRAYSCALE
    # =========================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mengurangi noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # =========================
    # FACE DETECTION
    # =========================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(120, 120)
    )

    # =========================
    # PROCESS EACH FACE
    # =========================

    for (x, y, w, h) in faces:

        # -------------------------
        # DRAW FACE
        # -------------------------

        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # -------------------------
        # EYE REGION
        # -------------------------

        eye_y1 = y + int(h * 0.15)
        eye_y2 = y + int(h * 0.55)

        eye_x1 = x
        eye_x2 = x + w

        # Pastikan koordinat valid
        eye_y1 = max(0, eye_y1)
        eye_y2 = min(gray.shape[0], eye_y2)

        eye_x1 = max(0, eye_x1)
        eye_x2 = min(gray.shape[1], eye_x2)

        eye_region_gray = gray[
            eye_y1:eye_y2,
            eye_x1:eye_x2
        ]

        eye_region_color = frame[
            eye_y1:eye_y2,
            eye_x1:eye_x2
        ]

        # -------------------------
        # EYE DETECTION
        # -------------------------

        eyes = eye_cascade.detectMultiScale(
            eye_region_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(25, 15),
            maxSize=(
                max(25, int(w * 0.45)),
                max(15, int(h * 0.25))
            )
        )

        # -------------------------
        # DRAW EYES
        # -------------------------

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                eye_region_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

            cv2.putText(
                eye_region_color,
                "Eye",
                (ex, max(15, ey - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # -------------------------
        # EYE COUNT
        # -------------------------

        cv2.putText(
            frame,
            f"Eyes: {len(eyes)}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # =========================
    # FACE COUNT
    # =========================

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =========================
    # DISPLAY
    # =========================

    cv2.imshow(
        "Drowsiness Detection - Face + Eye",
        frame
    )

    # =========================
    # QUIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()