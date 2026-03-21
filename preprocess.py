import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size
    resized = cv2.resize(gray, (64, 64))

    # Normalize pixel values
    normalized = resized / 255.0

    return normalized

img = preprocess_image("data/smiling/7.jpg")
print(img.shape, img.min(), img.max())
