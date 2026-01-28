import time
import cv2
import numpy as np
import os
import pandas as pd
import warnings
import torch
from ultralytics import YOLO, RTDETR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from test_data.test import images, labels

def get_device():
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()
print(f"🚀 Running on: {str(DEVICE).upper()}")

# --- CONFIGURATION ---
IMAGES_DIR = "./test_data/test/images"
LABELS_DIR = "./test_data/test/images"
OUTPUT_FILE = "benchmark_results_conf_01.txt"
OUTPUT_INFERENCE_DIR = "runs_benchmark_conf_01"

TENT_GROUP_LABELS = ["tent", "kite", "umbrella"]
PERSON_LABELS = ["person"]

MODELS_TO_TEST = [
    "yolov9c.pt",
    "yolov9s.pt",
    "rtdetr-l.pt",
    "yolo11m.pt",
    "yolo26s.pt",
    "yolo26l.pt"
]

# SAHI Settings
SLICE_H, SLICE_W = 640, 640
OVERLAP_RATIO = 0.2
CONF_THRES = 0.25
IOU_THRES = 0.50

# --- MATH & METRICS ---

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def xywhn2xyxy(x, w_img, h_img):
    xc, yc, w, h = x[0], x[1], x[2], x[3]
    x1 = int((xc - w / 2) * w_img)
    y1 = int((yc - h / 2) * h_img)
    x2 = int((xc + w / 2) * w_img)
    y2 = int((yc + h / 2) * h_img)
    return [x1, y1, x2, y2]

class MAPCalculator:
    def __init__(self):
        self.preds_person = []
        self.preds_tent = []
        self.total_gt_person = 0
        self.total_gt_tent = 0

    def update(self, pred_boxes, gt_boxes):
        pred_p = [p for p in pred_boxes if int(p[4]) == 0]
        pred_t = [p for p in pred_boxes if int(p[4]) == 1]
        gt_p = [g for g in gt_boxes if int(g[4]) == 0]
        gt_t = [g for g in gt_boxes if int(g[4]) == 1]
        
        self.total_gt_person += len(gt_p)
        self.total_gt_tent += len(gt_t)
        self._match_batch(pred_p, gt_p, self.preds_person)
        self._match_batch(pred_t, gt_t, self.preds_tent)

    def _match_batch(self, preds, gts, storage_list):
        preds.sort(key=lambda x: x[5], reverse=True)
        gt_matched = [False] * len(gts)
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            for idx, g in enumerate(gts):
                iou = compute_iou(p[:4], g[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= IOU_THRES and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                storage_list.append([p[5], 1])
                gt_matched[best_gt_idx] = True
            else:
                storage_list.append([p[5], 0])

    def compute_map(self):
        def calc_ap(preds, total_gt):
            if total_gt == 0 or not preds: return 0.0
            preds.sort(key=lambda x: x[0], reverse=True)
            tps = np.array([p[1] for p in preds])
            recalls = np.cumsum(tps) / total_gt
            precisions = np.cumsum(tps) / (np.arange(len(tps)) + 1)
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        ap_p = calc_ap(self.preds_person, self.total_gt_person)
        ap_t = calc_ap(self.preds_tent, self.total_gt_tent)
        return ap_p, ap_t, (ap_p + ap_t) / 2

# --- DATA & LOGIC ---

def get_model_specific_ids(names_dict, target_labels):
    found_ids = []
    name_to_id = {v.lower(): k for k, v in names_dict.items()}
    for label in target_labels:
        if label.lower() in name_to_id:
            found_ids.append(name_to_id[label.lower()])
    return found_ids

def load_ground_truth(filename, w, h, tent_ids, person_ids):
    path = os.path.join(LABELS_DIR, os.path.splitext(filename)[0] + ".txt")
    boxes = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                p = list(map(float, line.split()))
                cls_id = int(p[0])
                if cls_id in tent_ids: boxes.append(xywhn2xyxy(p[1:], w, h) + [1])
                elif cls_id in person_ids: boxes.append(xywhn2xyxy(p[1:], w, h) + [0])
    return boxes

def apply_winner_take_all(preds):
    people = [p for p in preds if p[4] == 0]
    tents = [p for p in preds if p[4] == 1]
    final = list(people)
    if tents:
        tents.sort(key=lambda x: x[5], reverse=True)
        final.append(tents[0])
    return final

def draw_and_save(img, preds, output_dir, filename, suffix):
    vis_img = img.copy()
    colors = {0: (0, 255, 0), 1: (255, 0, 0)}
    for p in preds:
        x1, y1, x2, y2 = map(int, p[:4])
        cls_id = int(p[4])
        color = colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, f"{cls_id} {p[5]:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{suffix}.jpg")
    cv2.imwrite(save_path, vis_img)

# --- INFERENCE ---

def run_standard(model, img_path, tent_ids, person_ids):
    # Timing happens outside this function to exclude function call overhead if strict, 
    # but here we include the prediction call.
    results = model.predict(img_path, conf=CONF_THRES, device=DEVICE, verbose=False)[0]
    preds = []
    if results.boxes:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in tent_ids or cls_id in person_ids:
                mapped_id = 1 if cls_id in tent_ids else 0
                preds.append(box.xyxy[0].cpu().numpy().tolist() + [mapped_id, box.conf[0].item()])
    return apply_winner_take_all(preds)

def run_sahi_prediction(detection_model, img_path, tent_ids, person_ids):
    # We pass the pre-loaded detection_model here
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        verbose=0
    )
    preds = []
    for obj in result.object_prediction_list:
        cls_id = obj.category.id
        if cls_id in tent_ids or cls_id in person_ids:
            mapped_id = 1 if cls_id in tent_ids else 0
            # SAHI returns [minx, miny, maxx, maxy] in bbox
            preds.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy, mapped_id, obj.score.value])
    return apply_winner_take_all(preds)

# --- MAIN BENCHMARK LOOP ---

def process_benchmarks():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    all_results = []
    os.makedirs(OUTPUT_INFERENCE_DIR, exist_ok=True)
    
    print(f"Starting Benchmark on {len(image_files)} images...")

    for model_path in MODELS_TO_TEST:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\n--- Processing {model_name} ---")
        model_run_dir = os.path.join(OUTPUT_INFERENCE_DIR, model_name)
        os.makedirs(model_run_dir, exist_ok=True)

        # 1. LOAD MODELS (Moved outside the image loop for speed!)
        print("   > Loading Standard Model...")
        std_model = RTDETR(model_path) if "rtdetr" in model_path.lower() else YOLO(model_path)
        
        # Get IDs
        tent_ids = get_model_specific_ids(std_model.names, TENT_GROUP_LABELS)
        person_ids = get_model_specific_ids(std_model.names, PERSON_LABELS)
        
        if not tent_ids and not person_ids:
            print(f"Skipping {model_name}: No target labels found.")
            continue
            
        print("   > Loading SAHI Model Wrapper...")
        m_type = "rtdetr" if "rtdetr" in model_path.lower() else "yolov8"
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type=m_type,
            model_path=model_path,
            confidence_threshold=CONF_THRES,
            device=DEVICE
        )

        calc_standard = MAPCalculator()
        calc_sahi = MAPCalculator()
        
        # Timers
        total_time_std = 0.0
        total_time_sahi = 0.0
        
        for img_file in image_files:
            img_path = os.path.join(IMAGES_DIR, img_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            gt_boxes = load_ground_truth(img_file, w, h, tent_ids, person_ids)
            
            # --- STANDARD RUN ---
            t0 = time.perf_counter()
            preds_std = run_standard(std_model, img_path, tent_ids, person_ids)
            t1 = time.perf_counter()
            total_time_std += (t1 - t0)
            
            calc_standard.update(preds_std, gt_boxes)
            draw_and_save(img, preds_std, model_run_dir, img_file, "std")
            
            # --- SAHI RUN ---
            t0 = time.perf_counter()
            preds_sahi = run_sahi_prediction(sahi_model, img_path, tent_ids, person_ids)
            t1 = time.perf_counter()
            total_time_sahi += (t1 - t0)
            
            calc_sahi.update(preds_sahi, gt_boxes)
            draw_and_save(img, preds_sahi, model_run_dir, img_file, "sahi")

        # Compute Metrics
        std_p, std_t, std_all = calc_standard.compute_map()
        sahi_p, sahi_t, sahi_all = calc_sahi.compute_map()
        
        # Compute Speed
        num_imgs = len(image_files)
        avg_std_ms = (total_time_std / num_imgs) * 1000
        fps_std = num_imgs / total_time_std if total_time_std > 0 else 0
        
        avg_sahi_ms = (total_time_sahi / num_imgs) * 1000
        fps_sahi = num_imgs / total_time_sahi if total_time_sahi > 0 else 0

        res = {
            "Model": model_name,
            # Accuracy
            "STD_mAP": round(std_all, 3),
            "SAHI_mAP": round(sahi_all, 3),
            # Speed Standard
            "STD_FPS": round(fps_std, 1),
            "STD_Lat(ms)": round(avg_std_ms, 1),
            # Speed SAHI
            "SAHI_FPS": round(fps_sahi, 1),
            "SAHI_Lat(ms)": round(avg_sahi_ms, 1),
        }
        all_results.append(res)
        print(f"Finished {model_name}: STD FPS={fps_std:.1f} | SAHI FPS={fps_sahi:.1f}")

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*80)
        print(" FINAL BENCHMARK RESULTS (Jetson Ready) ")
        print("="*80)
        print(df.to_string())
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(df.to_string())
        print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_benchmarks()
