import io
import os
import glob
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
        combined = sv.Detections.merge(all_detections)

        # Apply NMS in-place and return it
        # iou_threshold defaults to 0.5 if you omit it
        final_detections = combined.with_nms(threshold=0.45)
        return final_detections
    else:
        return sv.Detections.empty()

def process_directory(input_dir, output_dir, model, tile_size=728, overlap=200, threshold=0.5):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save annotated images
        model: Model to use for detection
        tile_size: Size of tiles
        overlap: Overlap between tiles
        threshold: Detection confidence threshold
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get filename without path and extension
        filename = os.path.basename(img_path)
        base_filename, _ = os.path.splitext(filename)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Process image
        detections = process_image_with_tiles(model, image, tile_size=tile_size, 
                                             overlap=overlap, threshold=threshold)
        
        # Create labels
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        
        # Annotate image
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"{base_filename}_annotated.jpg")
        annotated_image.save(output_path)
        
    print(f"All images processed and saved to {output_dir}")

# Initialize the model
model = RFDETRLarge(resolution=728)

# Option 1: Process a single image (original code)
def process_single_image():
    image = Image.open("data/b_50_frames/frame_000960.jpg")
    detections = process_image_with_tiles(model, image, tile_size=728, overlap=200, threshold=0.5)
    
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    
    annotated_image.save("output.jpg")
    sv.plot_image(annotated_image)

# Option 2: Process all images in a directory
if __name__ == "__main__":
    import sys
    
    # Default directories
    input_dir = "data/b_50_frames"
    output_dir = "output"
    
    # Parse command-line arguments if provided
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Process the directory of images
    print(f"Processing all images in {input_dir}")
    process_directory(input_dir, output_dir, model)
    
    # Uncomment to process a single image instead
    # process_single_image()