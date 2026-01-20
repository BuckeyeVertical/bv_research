from PIL import Image
import os
import cv2
import numpy as np

PADDING = 70


def run(data):
    image = data["image"]
    label_content = data["label_content"]

    if not label_content:
        return data

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    max_h, max_w = image_cv.shape[:2]

    mannequins = []
    lines = label_content.strip().splitlines()

    for line in lines:
        coords = line.strip().split()
        if len(coords) < 5:
            continue
        center_x_norm, center_y_norm, w_norm, h_norm = map(float, coords[1:5])

        w = int(w_norm * max_w)
        h = int(h_norm * max_h)
        center_x = int(center_x_norm * max_w)
        center_y = int(center_y_norm * max_h)

        origin_x = int(center_x - w/2)
        origin_y = int(center_y - h/2)

        x1 = origin_x - PADDING
        x2 = origin_x + w + PADDING
        y1 = origin_y - PADDING
        y2 = origin_y + h + PADDING

        x1 = max(0, x1)
        x2 = min(max_w, x2)
        y1 = max(0, y1)
        y2 = min(max_h, y2)

        cropped_cv = image_cv[y1:y2, x1:x2]

        mannequins.append({
            "crop": cropped_cv,
            "center": (center_x, center_y),
        })

    data["mannequins"] = mannequins
    print(image)
    return data
