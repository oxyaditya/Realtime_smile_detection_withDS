import cv2
import joblib
import numpy as np
from preprocess import preprocess_image

# Load trained Logistic Regression model
model = joblib.load("lr_smile_model.pkl")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam (IMPORTANT: CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera opened:", cap.isOpened())

# Create window explicitly (IMPORTANT)
cv2.namedWindow("Live Smile Detection", cv2.WINDOW_NORMAL)

print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y + h//2 : y + h, x : x + w]


        # Save face temporarily (simple & safe)
        cv2.imwrite("temp_face.jpg", face)

        # Preprocess
        img = preprocess_image("temp_face.jpg")
        img = img.reshape(1, -1)

        # Predict
        pred = model.predict(img)[0]
        prob = model.predict_proba(img)[0]

        label = "SMILING " if pred == 1 else "NOT SMILING"
        confidence = max(prob)

        # Draw box & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Smile Detection", frame)

    key = cv2.waitKey(30)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
