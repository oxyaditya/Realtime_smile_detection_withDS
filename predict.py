import numpy as np
import cv2
from preprocess import preprocess_image
from dataset import load_dataset
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_dataset("data")

# Flatten images
X = X.reshape(X.shape[0], -1)

# Train model on FULL dataset
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------- PREDICT NEW IMAGE --------
image_path = "DATA/test/Aaron_Peirsol_0002.jpg"   # change this path

img = preprocess_image(image_path)
img = img.reshape(1, -1)   # flatten

prediction = model.predict(img)[0]
probability = model.predict_proba(img)[0]

if prediction == 1:
    print("Prediction: SMILING 😄")
else:
    print("Prediction: NOT SMILING 😐")

print("Confidence:", probability)
