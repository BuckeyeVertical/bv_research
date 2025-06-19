import io
import os
import glob
import requests
import supervision as sv
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from torchvision import transforms

class Detector():
    def __init__(self, batch_size=4, resolution=728):
        self.batch_size = batch_size
        self.resolution = resolution
        self.model = self.create_model()
        self.frame_save_cnt = 0

    def create_tiles(self, image, overlap=100):
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
        tile_size = self.resolution
        width, height = image.shape[1], image.shape[0]
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
                tile = image[y_start:y_end, x_start:x_end, :]
                tiles.append((tile, (x_start, y_start)))
        
        return tiles

    def pad_and_predict(self, tiles, threshold):
        # tiles: list of H×W×C numpy arrays (or torch tensors)
        # model.predict wants exactly `batch_size` inputs.

        # 1) prepare batch
        B = len(tiles)
        if B < self.batch_size:
            # assume tiles[0] exists; make dummy using its shape
            dummy = np.zeros_like(tiles[0])
            tiles = tiles + [dummy] * (self.batch_size - B)
            trim_to = B
        else:
            trim_to = None

        # 2) run inference
        outputs = self.model.predict(tiles, threshold=threshold)

        # 3) if we padded, slice off the extra outputs
        if trim_to is not None:
            # assume outputs is a list of detections per‐tile
            outputs = outputs[:trim_to]

        return outputs

    def process_frame(self, frame, overlap: int = 100, threshold: float = 0.5):
        """
        Process an image by tiling, using RF-DETR's batch API for inference,
        then combine results and apply global NMS.
        """
        # 1) Tile the image and unzip into lists of tiles + offsets
        tiles_with_positions = self.create_tiles(frame, overlap)
        tiles, offsets = zip(*tiles_with_positions)

        if self.model == None:
            print("Creating model")
            self.model = self.create_model()

        detections_list = []

        for i in range(0, len(tiles), self.batch_size):
            batch = list(tiles[i:i + self.batch_size])
            dets = self.pad_and_predict(batch, threshold)
            detections_list.extend(dets)

        # 3) Adjust boxes back to full-image coords
        all_dets = []
        for dets, (x_off, y_off) in zip(detections_list, offsets):
            if len(dets.xyxy) == 0:
                continue

            boxes = dets.xyxy.copy()               # NumPy array copy
            boxes[:, [0, 2]] += x_off              # x_min, x_max
            boxes[:, [1, 3]] += y_off              # y_min, y_max

            all_dets.append(sv.Detections(
                xyxy=boxes,
                confidence=dets.confidence,
                class_id=dets.class_id
            ))

        # 4) Merge & run global NMS
        if not all_dets:
            return sv.Detections.empty()

        merged = sv.Detections.merge(all_dets)
        return merged.with_nms(threshold=0.45)
    
    def annotate_frame(self, frame, detections, class_names=COCO_CLASSES):
        """
        Annotate `frame` with bounding boxes and per-detection labels + confidence.

        Args:
            frame: H×W×C NumPy array
            detections: sv.Detections object containing `xyxy`, `class_id`, `confidence`, etc.
            class_names: list of class names (e.g. COCO_CLASSES)

        Returns:
            Annotated image as NumPy array
        """
        annotated_frame = frame.copy()
        # Draw bounding boxes
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        # Prepare labels like 'class 0.85'
        labels = [f"{class_names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
        # Draw class labels with confidence
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
        return annotated_frame

    def save_frame(self, frame, output_dir):
        output_path = os.path.join(output_dir, f"{self.frame_save_cnt}_annotated.jpg")
        Image.fromarray(frame).save(output_path)
        self.frame_save_cnt += 1

    def create_model(self, dtype=torch.float16):
        # Initialize the model
        torch.cuda.empty_cache()
        model = RFDETRBase(resolution=self.resolution)

        model.optimize_for_inference(batch_size=self.batch_size, dtype=dtype)

        print(f"Model created with batch size: {self.batch_size}")

        return model
    
    def process_directory(self, input_dir, output_dir,
                           overlap: int = 100,
                           threshold: float = 0.5,
                           labels=COCO_CLASSES):
        """
        Process all images in `input_dir`, save annotated outputs to `output_dir`.
        Supports .jpg, .jpeg, .png files and shows a progress bar.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Gather image file paths
        patterns = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for pat in patterns:
            image_paths.extend(glob.glob(os.path.join(input_dir, pat)))
        image_paths.sort()

        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Load and convert to NumPy array
            pil_img = Image.open(img_path).convert("RGB")
            frame = np.array(pil_img)

            # Run detection
            detections = self.process_frame(frame, overlap, threshold)

            # Annotate
            annotated = self.annotate_frame(frame, detections, labels)

            # Save result
            self.save_frame(annotated, output_dir)

# Option 2: Process all images in a directory
if __name__ == "__main__":
    import sys
    
    # Default directories
    input_dir = "data/b_70_frames"
    output_dir = "output/b_70"
    
    # Parse command-line arguments if provided
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Process the directory of images
    print(f"Processing all images in {input_dir}")

    detector = Detector()
    detector.process_directory(input_dir, output_dir)
    
    # Uncomment to process a single image instead
    # process_single_image()