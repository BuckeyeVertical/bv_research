import cv2
import numpy as np
from ultralytics import YOLO
modelname = 'yolo12m.pt'  # Change to 'yolov8n.pt' for detection-only, 'yolov8n-seg.pt' for segmentation
# Load the YOLO model
model = YOLO(modelname)  # Use 'yolov8n.pt' for detection-only, 'yolov8n-seg.pt' for segmentation

def split_frame_with_padding(frame, tile_size=(512, 512), padding=32):
    h, w, _ = frame.shape
    tile_h, tile_w = tile_size
    tiles = []

    for y in range(0, h, tile_h - 2 * padding):
        for x in range(0, w, tile_w - 2 * padding):
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + tile_w + padding, w)
            y_end = min(y + tile_h + padding, h)

            tile = frame[y_start:y_end, x_start:x_end]
            tiles.append((tile, (x_start, y_start)))

    return tiles

def detect_objects_in_tile(tile):
    """Run YOLO detection on a tile and return tile with drawn boxes and labels."""
    results = model.predict(tile, verbose=False)[0]
    annotated_tile = tile.copy()

    if results.masks is not None:
        # If segmentation masks are available
        masks = results.masks.xy
        for mask in masks:
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(annotated_tile, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(annotated_tile, [points], color=(0, 255, 0, 50))  # translucent mask

    if results.boxes is not None:
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            confidence = float(conf)

            # Get label name
            label = model.names[class_id] if hasattr(model, "names") else str(class_id)
            label_text = f"{label} {confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(annotated_tile, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Put label above the box
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_tile, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 0, 0), -1)
            cv2.putText(annotated_tile, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_tile

  

def process_video(input_path, output_path=None, tile_size=(512, 512), padding=32):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = np.zeros_like(frame)

        tiles = split_frame_with_padding(frame, tile_size=tile_size, padding=padding)

        for tile, (x_start, y_start) in tiles:
            processed_tile = detect_objects_in_tile(tile)

            tile_h, tile_w, _ = processed_tile.shape

            x1 = x_start + padding if x_start != 0 else x_start
            y1 = y_start + padding if y_start != 0 else y_start
            x2 = x_start + tile_w - padding if (x_start + tile_w) < width else x_start + tile_w
            y2 = y_start + tile_h - padding if (y_start + tile_h) < height else y_start + tile_h

            output_frame[y1:y2, x1:x2] = processed_tile[
                (y1 - y_start):(y2 - y_start),
                (x1 - x_start):(x2 - x_start)
            ]

        if out:
            out.write(output_frame)

    cap.release()
    if out:
        out.release()
if __name__ == "__main__":
    input_video = "videos/DJI_0029.mp4"      # Change this to your video path
    output_video = f"{modelname}output.mp4"     # Change or set to None if you don't want to save
    process_video(input_video, output_video)
