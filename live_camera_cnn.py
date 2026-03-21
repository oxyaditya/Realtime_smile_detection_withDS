import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("cnn_smile_model.keras")
print("CNN model loaded")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Smile Detection (CNN)", cv2.WINDOW_NORMAL)

last_face = None   # store last detected face
label = "NO FACE"
prob = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(120, 120)
    )

    # If a face is detected, update last_face
    if len(faces) > 0:
        # Take the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        last_face = (x, y, w, h)

    # Use last detected face
    if last_face is not None:
        x, y, w, h = last_face
        face = frame[y:y+h, x:x+w]

        img = cv2.resize(face, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.reshape(1, 64, 64, 1)

        # Predict every few frames
        if frame_count % 4 == 0:
            prob = model.predict(img, verbose=0)[0][0]
            label = "SMILING" if prob > 0.4 else "NOT SMILING"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    else:
        label = "FACE NOT DETECTED"

    # Draw label
    cv2.putText(
        frame,
        f"{label} ({prob:.2f})",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0) if "SMILING" in label else (0,0,255),
        2
    )

    cv2.imshow("Smile Detection (CNN)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
