import io
import requests
import supervision as sv
import numpy as np
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from tqdm import tqdm

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

def create_tiles(image, tile_size=728, overlap=100):
    """
    Split an image into tiles with overlap.
    
    Args:
        image: PIL Image to be tiled
        tile_size: Size of each square tile
        overlap: Overlap between adjacent tiles in pixels
    
    Returns:
        List of (tile, (x_offset, y_offset)) where tile is a PIL Image and
        (x_offset, y_offset) is the position of the tile in the original image
    """
    width, height = image.size
    tiles = []
    
    # Calculate positions where tiles should start
    x_positions = list(range(0, width - overlap, tile_size - overlap))
    if width not in x_positions and x_positions:
        x_positions.append(max(0, width - tile_size))
    if not x_positions:  # Image is smaller than tile_size
        x_positions = [0]
        
    y_positions = list(range(0, height - overlap, tile_size - overlap))
    if height not in y_positions and y_positions:
        y_positions.append(max(0, height - tile_size))
    if not y_positions:  # Image is smaller than tile_size
        y_positions = [0]
    
    # Create tiles
    for y in y_positions:
        for x in x_positions:
            # Adjust position if we're at the edge of the image
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            # Extract tile
            tile = image.crop((x_start, y_start, x_end, y_end))
            tiles.append((tile, (x_start, y_start)))
    
    return tiles

def process_image_with_tiles(model, image, tile_size=728, overlap=100, threshold=0.5):
    """
    Process an image by tiling it and running the model on each tile.
    Then combine the results and apply NMS.
    """
    # Create tiles
    tiles_with_positions = create_tiles(image, tile_size, overlap)
    
    all_detections = []
    
    # Process each tile with progress bar
    for tile, (x_offset, y_offset) in tqdm(tiles_with_positions, desc="Processing tiles"):
        # Run model on tile
        tile_detections = model.predict(tile, threshold=threshold)
        
        if len(tile_detections.xyxy) > 0:
            # Adjust bounding box coordinates to the original image
            adjusted_boxes = tile_detections.xyxy.copy()
            adjusted_boxes[:, 0] += x_offset  # x_min
            adjusted_boxes[:, 1] += y_offset  # y_min
            adjusted_boxes[:, 2] += x_offset  # x_max
            adjusted_boxes[:, 3] += y_offset  # y_max
            
            # Append to all detections
            all_detections.append(sv.Detections(
                xyxy=adjusted_boxes,
                confidence=tile_detections.confidence,
                class_id=tile_detections.class_id
            ))
    
    # Combine all detections
    if all_detections:
        combined_detections = sv.Detections.merge(all_detections)
        
        # Apply NMS to remove duplicate detections
        nms_detections = sv.box_utils.non_max_suppression(
            boxes=combined_detections.xyxy,
            scores=combined_detections.confidence,
            iou_threshold=0.45,  # Adjust as needed
            class_ids=combined_detections.class_id
        )
        
        return sv.Detections(
            xyxy=nms_detections[0],
            confidence=nms_detections[1],
            class_id=nms_detections[2]
        )
    else:
        # Return empty detections if no objects found
        return sv.Detections.empty()

model = RFDETRLarge(resolution=3808)

image = Image.open("data/b_50_frames/frame_000960.jpg").convert("RGB")
# Remove the undefined crop variable check
# Process with tiling directly
detections = process_image_with_tiles(model, image, tile_size=728, overlap=200, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)