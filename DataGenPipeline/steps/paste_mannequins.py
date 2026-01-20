import cv2
import numpy as np
import os
from PIL import Image


def run(data):
    image = data["image"]
    mannequins = data.get("mannequins", [])

    if not mannequins:
        return data

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    dest_h, dest_w = image_cv.shape[:2]

    for mannequin in mannequins:
        crop_cv = mannequin["crop"]
        center = mannequin["center"]

        mask = 255 * np.ones(crop_cv.shape, crop_cv.dtype)

        try:
            image_cv = cv2.seamlessClone(
                crop_cv,
                image_cv,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
        except Exception as e:
            print(f"Error pasting mannequin at {center}: {e}")

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    data["image"] = Image.fromarray(image_rgb)

    return data
