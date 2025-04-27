import cv2
import numpy as np
from ultralytics import YOLO

modelname = 'yolo12m.pt'
model = YOLO(modelname)

def split_frame_with_padding(frame, tile_size=(512, 512), padding=32):
    h, w, _ = frame.shape
    tile_h, tile_w = tile_size
    tiles = []
    for y in range(0, h, tile_h - 2 * padding):
        for x in range(0, w, tile_w - 2 * padding):
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end   = min(x + tile_w + padding, w)
            y_end   = min(y + tile_h + padding, h)
            tile = frame[y_start:y_end, x_start:x_end]
            tiles.append((tile, x_start, y_start))
    return tiles

def detect_on_frame(frame, tile_size=(512,512), padding=32):
    """Runs YOLO on all tiles, returns a list of detections in global coords."""
    detections = []  # each entry: dict with x1,y1,x2,y2,conf,class_id,mask_polygons
    tiles = split_frame_with_padding(frame, tile_size, padding)
    for tile, x_off, y_off in tiles:
        results = model.predict(tile, verbose=False)[0]
        # ---- segmentation masks ----
        if hasattr(results, 'masks') and results.masks is not None:
            for mask_xy in results.masks.xy:
                # mask_xy is an (N,2) array of polygon vertices in tile coords
                pts = np.array(mask_xy, dtype=np.int32)
                pts[:, 0] += x_off
                pts[:, 1] += y_off
                detections.append({'mask': pts})
        # ---- bounding boxes ----
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 + x_off), int(y1 + y_off), int(x2 + x_off), int(y2 + y_off)
            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': float(conf),
                'class_id': int(cls),
            })
    return detections

def draw_detections(frame, detections):
    """Draws boxes, labels, and masks on frame in-place."""
    for det in detections:
        if 'mask' in det:
            cv2.polylines(frame, [det['mask']], isClosed=True, color=(0,255,0), thickness=2)
            cv2.fillPoly(frame, [det['mask']], color=(0,255,0,50))
        if 'box' in det:
            x1, y1, x2, y2 = det['box']
            label = model.names[det['class_id']] if hasattr(model, "names") else str(det['class_id'])
            text = f"{label} {det['conf']:.2f}"
            # box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            # label bg
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (255,0,0), -1)
            cv2.putText(frame, text, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def process_video(input_path, output_path=None, tile_size=(512,512), padding=32, skip=20):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open {input_path}")
        return
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # every `skip` frames, re-run detection; otherwise reuse
        if frame_idx % skip == 0:
            last_detections = detect_on_frame(frame, tile_size, padding)

        # draw detections on a copy of the fresh frame
        annotated = frame.copy()
        annotated = draw_detections(annotated, last_detections)

        if out:
            out.write(annotated)

    cap.release()
    if out:
        out.release()

if __name__ == "__main__":
    input_video  = "videos/DJI_0029.mp4"
    output_video = f"{modelname}_every20.mp4"
    process_video(input_video, output_video)
