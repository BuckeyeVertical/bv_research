import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRLarge(resolution=3808)

image = Image.open("data/b_50_frames/frame_000960.jpg")
image = image.resize((3808, 3808), Image.BILINEAR)
image = image.convert("RGB")
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