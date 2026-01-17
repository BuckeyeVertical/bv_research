import cv2
import os


def load(file_path):
    image = cv2.imread(file_path)

    if image is None:
        raise ValueError("path not valid")

    name = os.path.basename(file_path)

    return {
        "name": name,
        "image": image
    }


def save(data):
    name = data["name"]
    image = data["image"]
    out_path = os.path.join("processed", name)
    os.makedirs("processed", exist_ok=True)
    cv2.imwrite(out_path, image)
    return out_path
