import os
import numpy as np
from preprocess import preprocess_image


def load_dataset(data_path):
    images = []
    labels = []

    # Smiling images → label 1
    smiling_path = os.path.join(data_path, "smiling")
    for file_name in os.listdir(smiling_path):
        img_path = os.path.join(smiling_path, file_name)
        try:
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(1)
        except:
            continue

    # Not smiling images → label 0
    not_smiling_path = os.path.join(data_path, "not_smiling")
    for file_name in os.listdir(not_smiling_path):
        img_path = os.path.join(not_smiling_path, file_name)
        try:
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(0)
        except:
            continue

    return np.array(images), np.array(labels)


# RUN THIS FILE
if __name__ == "__main__":
    X, y = load_dataset("data")

    print("Total images:", len(X))
    print("Smiling images:", np.sum(y == 1))
    print("Not smiling images:", np.sum(y == 0))

    print("First 30 labels:", y[:30])
    print("Last 30 labels:", y[-30:])
