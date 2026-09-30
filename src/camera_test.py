import cv2

# =========================
# LOAD FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Gagal memuat face cascade.")
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

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sedikit blur untuk mengurangi noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # =========================
    # FACE DETECTION
    # =========================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(150, 150)
    )

    # =========================
    # DRAW FACE
    # =========================

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # =========================
    # INFORMATION
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
        "Drowsiness Detection - Face Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()