import numpy as np
from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data
X, y = load_dataset("data")

# Flatten images (64x64 → 4096)
X = X.reshape(X.shape[0], -1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import joblib

joblib.dump(model, "lr_smile_model.pkl")
print("Saved Logistic Regression model.")
