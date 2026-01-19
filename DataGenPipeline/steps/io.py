import cv2
import os
from PIL import Image

class AlreadyProcessedError(Exception):
    pass

def load(file_path):
    name = os.path.basename(file_path)
    if os.path.exists(os.path.join("processed", name)):
        raise ValueError("image already processed")

    image = Image.open(file_path)
    if image is None:
        raise ValueError("path not valid")

    return {
        "name": name,
        "image": image
    }


def save(data):
    name = data["name"]
    print(f"save {name}")
    image = data["image"]
    out_path = os.path.join("processed", name)
    os.makedirs("processed", exist_ok=True)
    image.save(out_path)
    return out_path
