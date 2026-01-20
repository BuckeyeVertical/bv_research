import cv2
import os
from PIL import Image

class AlreadyProcessedError(Exception):
    pass

def load(image_path, label_path):
    name = os.path.basename(image_path)
    if os.path.exists(os.path.join("processed", name)):
        raise ValueError("image already processed")
    
    image = Image.open(image_path)
    if image is None:
        raise ValueError("path not valid")

    with open(label_path, "r") as f:
        label_content = f.read()
    print(label_content)
    print(image)
    return {
        "name": name,
        "image": image,
        "label_content": label_content,
        "mannequins": []
    }

def save(data):
    name = data["name"]
    print(f"save {name}")
    image = data["image"]
    out_path = os.path.join("processed", name)
    os.makedirs("processed", exist_ok=True)
    image.save(out_path)
    return out_path
