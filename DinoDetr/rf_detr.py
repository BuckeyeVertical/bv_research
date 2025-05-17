import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

def letterbox_image(img: Image.Image, target_size: int, fill_color=(114,114,114)):
    """
    Resize an image to fit within a square of size (target_size x target_size) 
    without changing aspect ratio, then pad the rest with `fill_color`.
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 1) Resize with preserved aspect ratio
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # 2) Create padded canvas
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    
    # 3) Paste resized image centered
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas.paste(img_resized, (left, top))
    
    return canvas

model = RFDETRLarge(resolution=3808)

image = Image.open("data/b_50_frames/frame_000960.jpg")
image = letterbox_image(image, target_size=3808)
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)