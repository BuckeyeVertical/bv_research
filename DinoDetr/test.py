#!/usr/bin/env python
"""
Test script for DINOv2 + DETR via HuggingFace Transformers, with tiling support.
"""

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class DINOv2Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')         
        self.out_channels = self.body.embed_dim

    def forward(self, x):
        # x: [B, 3, H, W]
        # get the raw transformer tokens:
        tokens = self.body.forward_features(x)    # [B, 1 + N, C]
        # drop the class token:
        tokens = tokens[:, 1:, :]                # [B, N, C]
        B, N, C = tokens.shape
        # assume square grid of patches:
        S = int(N**0.5)
        # reshape into [B, C, S, S]
        feat_map = tokens.permute(0, 2, 1).view(B, C, S, S)
        return {'0': feat_map}

def tile_image(img, tile_size=800, overlap=100):
    w, h = img.size
    stride = tile_size - overlap
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile_size, h)
            right  = min(left + tile_size, w)
            tile = img.crop((left, top, right, bottom))
            yield tile, (left, top)

# Reuse your normalization pipeline
transform_tile = Compose([
    Resize(800),  
    ToTensor(),
    Normalize(mean=MEAN, std=STD),
])

def run_inference_on_tiles(model, feature_extractor, img, device, thresh=0.7):
    all_bboxes, all_scores, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for tile, (lx, ly) in tile_image(img):
            # 1) Prepare inputs
            inputs = feature_extractor(images=tile, return_tensors="pt").to(device)
            # 2) Forward
            outputs = model(**inputs)
            # 3) Post-process to get boxes in pixel coords
            result = feature_extractor.post_process_object_detection(
                outputs, threshold=thresh, target_sizes=[tile.size[::-1]]
            )[0]
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            # 4) Shift back to original image coords
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = box
                all_bboxes.append([x0+lx, y0+ly, x1+lx, y1+ly])
                all_scores.append(score.item())
                all_labels.append(label.item())

    # 5) NMS per class
    final_boxes, final_scores, final_labels = [], [], []
    if all_bboxes:
        b = torch.tensor(all_bboxes)
        s = torch.tensor(all_scores)
        l = torch.tensor(all_labels)
        for cls in l.unique():
            idx = (l == cls).nonzero(as_tuple=True)[0]
            keep = nms(b[idx], s[idx], iou_threshold=0.5)
            for i in keep:
                final_boxes.append(b[idx[i]].tolist())
                final_scores.append(s[idx[i]].item())
                final_labels.append(cls.item())
    return final_boxes, final_scores, final_labels

def visualize(img, boxes, scores, labels):
    fig, ax = plt.subplots(1, figsize=(12,12))
    ax.imshow(img)
    for (x0,y0,x1,y1), scr, lbl in zip(boxes, scores, labels):
        rect = patches.Rectangle((x0,y0), x1-x0, y1-y0, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0, f"{lbl}:{scr:.2f}", fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def main(image_path='test.jpg', thresh=0.7, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 1) Load DINOv2 backbone
    # backbone = DINOv2Backbone().to(device)

    # 2) Prepare a DETR config and override its backbone to use DINOv2
    config = DetrConfig.from_pretrained('facebook/detr-resnet-50')
    config.num_labels = 91  # COCO classes
    # config.hidden_size = backbone.out_channels
    model = DetrForObjectDetection(config).from_pretrained('facebook/detr-resnet-50').to(device)

    # 3) Monkey-patch the DETR backbone
    # model.model.backbone.body = backbone
    # (Depending on HF version, you may also need to adjust norm layers etc.)

    # 4) Load a feature extractor for preprocessing & postprocessing
    fe = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    # 5) Run tile-based inference
    img = Image.open(image_path).convert('RGB')
    boxes, scores, labels = run_inference_on_tiles(model, fe, img, device, thresh)

    visualize(img, boxes, scores, labels)

if __name__ == '__main__':
    main(image_path="/home/eashan/workspace/bv2425ObjectDetection/data/b_50_frames/frame_000720.jpg")
