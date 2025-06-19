import os
import glob
from PIL import Image
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO  # pip install ultralytics

class Detector():
    def __init__(self, batch_size=256, tile_size=640, overlap=192, conf=0.5, iou=0.3):
        self.model = None
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf = conf
        self.iou = iou

    def letterbox_image(self, img: Image.Image, target_size: int, fill_color=(114,114,114)):
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

    def create_tiles(self, image, tile_size=336, overlap=100):
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

    def create_model(self):
        # Load your pretrained YOLOv12m
        model = YOLO("yolo12m.pt")
        model.fuse()                     # fuse Conv+BN for speed
        model.to("cuda" if model.device.type == "cuda" else "cpu")
        return model

    def process_image_with_tiles_batched(self, image: Image.Image):
        # 1) Tile
        tiles_with_pos = self.create_tiles(image, self.tile_size, self.overlap)
        tiles, offsets = zip(*tiles_with_pos)

        # 2) Lazy‐load model
        if self.model is None:
            self.model = self.create_model()

        detections_list = []
        print(f"Tile count: {len(tiles)}")
        # 3) Run in batches
        for i in range(0, len(tiles), self.batch_size):
            print("Entering loop")
            batch_tiles = tiles[i : min(i + self.batch_size, len(tiles))]
            # convert PIL→ndarray
            batch_np = [np.array(t) for t in batch_tiles]

            # Ultralytics batch inference:
            results = self.model.predict(
                source=batch_np,
                imgsz=self.tile_size,
                batch=self.batch_size,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )
            # results is a list of ulralytics.engine.results.Results
            detections_list.extend(results)

        # 4) Re‐assemble into supervision.Detections
        all_dets = []
        for res, (x_off, y_off) in zip(detections_list, offsets):
            # .boxes.xyxy is (N,4) tensor on device
            xyxy = res.boxes.xyxy.cpu().numpy()
            if xyxy.size == 0:
                continue

            # shift back to full‐image coords
            xyxy[:, [0,2]] += x_off
            xyxy[:, [1,3]] += y_off

            all_dets.append(
                sv.Detections(
                    xyxy=xyxy,
                    confidence=res.boxes.conf.cpu().numpy(),
                    class_id=res.boxes.cls.cpu().numpy().astype(int),
                )
            )

        # 5) Merge + global NMS
        if not all_dets:
            return sv.Detections.empty()

        merged = sv.Detections.merge(all_dets)
        return merged.with_nms(threshold=self.iou)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_files = []
        for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.tiff"]:
            image_files += glob.glob(os.path.join(input_dir, ext))
        print(f"Found {len(image_files)} images")

        for path in tqdm(image_files, desc="Images"):
            im = Image.open(path).convert("RGB")
            dets = self.process_image_with_tiles_batched(im)

            # annotate & save
            labels = [f"{d} {c:.2f}" for d,c in zip(dets.class_id, dets.confidence)]
            ann = sv.BoxAnnotator().annotate(im, dets)
            ann = sv.LabelAnnotator().annotate(ann, dets, labels)

            base = os.path.splitext(os.path.basename(path))[0]
            ann.save(os.path.join(output_dir, f"{base}_yolo12m.jpg"))


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else "data/b_70_frames"
    out = sys.argv[2] if len(sys.argv)>2 else "output/b_70_yolo12m"
    Detector().process_directory(inp, out)
