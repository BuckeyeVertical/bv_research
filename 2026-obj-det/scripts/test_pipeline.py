import os
import json
import numpy as np
from PIL import Image
import torch
from sahi.predict import get_sliced_prediction, get_prediction
from sahi import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================================
# 1. CUSTOM SAHI WRAPPER FOR LT-DETR
# ==========================================
class LightlyTrainDetectionModel(DetectionModel):
    """
    SAHI-compatible wrapper for LightlyTrain (LT-DETR) models.

    Per the LightlyTrain docs:
    - model.predict() accepts file paths, URLs, PIL Images, or tensors
    - Returns dict: {"labels": tensor(num_boxes,), "bboxes": tensor(num_boxes,4) in xyxy, "scores": tensor(num_boxes,)}
    - model.classes is a list of class names indexed by label ID
    """

    def load_model(self):
        import lightly_train
        self.model = lightly_train.load_model(self.model_path)

        # Move model to the correct device
        if self.device is not None:
            self.model = self.model.to(self.device)

        # Build category_mapping from model.classes
        if hasattr(self.model, "classes") and self.model.classes is not None:
            classes = self.model.classes
            if isinstance(classes, dict):
                # e.g. {0: "person", 1: "car", ...}
                self.category_names = [classes[k] for k in sorted(classes.keys())]
                self.category_mapping = {
                    str(k): str(v) for k, v in classes.items()
                }
            elif isinstance(classes, (list, tuple)):
                self.category_names = list(classes)
                self.category_mapping = {
                    str(i): name for i, name in enumerate(classes)
                }
            else:
                self.category_mapping = {str(i): str(i) for i in range(1000)}
                self.category_names = [str(i) for i in range(1000)]
        else:
            self.category_mapping = {str(i): str(i) for i in range(1000)}
            self.category_names = [str(i) for i in range(1000)]

        self.num_categories = len(self.category_mapping)

    def perform_inference(self, image: np.ndarray):
        """
        SAHI passes numpy arrays in RGB order.
        LightlyTrain predict() accepts PIL Images.
        """
        pil_img = Image.fromarray(image)
        self._original_predictions = self.model.predict(pil_img)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        original_predictions = self._original_predictions

        # Use SAHI compatibility helpers for safe defaults
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        shift_amount = shift_amount_list[0]
        full_shape = full_shape_list[0]

        object_prediction_list = []

        bboxes = original_predictions["bboxes"].cpu().numpy()
        scores = original_predictions["scores"].cpu().numpy()
        labels = original_predictions["labels"].cpu().numpy()

        for bbox, score, label in zip(bboxes, scores, labels):
            if score < self.confidence_threshold:
                continue

            xmin, ymin, xmax, ymax = bbox.tolist()

            # Validate bbox
            if xmin >= xmax or ymin >= ymax:
                continue

            category_id = int(label)
            category_name = self.category_mapping.get(
                str(category_id), str(category_id)
            )

            object_prediction = ObjectPrediction(
                bbox=[xmin, ymin, xmax, ymax],
                category_id=category_id,
                score=float(score),
                bool_mask=None,
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        return object_prediction_list


# ==========================================
# 2. CATEGORY ID MAPPING HELPER
# ==========================================
def build_category_id_remap(coco_gt, detection_model, model_type):
    """
    Build pred_id -> gt_id mapping by matching class names.
    This is the #1 cause of near-zero AP.
    """
    gt_cats = coco_gt.cats
    gt_name_to_id = {}
    for cat_id, cat_info in gt_cats.items():
        name = cat_info["name"].strip().lower()
        gt_name_to_id[name] = cat_id

    print(f"  Ground truth categories: {gt_name_to_id}")

    model_name_to_pred_id = {}
    if model_type == "ultralytics":
        try:
            names = detection_model.model.model.names
            if isinstance(names, dict):
                for idx, name in names.items():
                    model_name_to_pred_id[name.strip().lower()] = int(idx)
            elif isinstance(names, list):
                for idx, name in enumerate(names):
                    model_name_to_pred_id[name.strip().lower()] = idx
            print(f"  Model categories: {model_name_to_pred_id}")
        except Exception as e:
            print(f"  Warning: Could not read model class names: {e}")
            return None
    elif model_type == "lightly_train":
        if hasattr(detection_model, 'category_mapping') and detection_model.category_mapping:
            for idx_str, name in detection_model.category_mapping.items():
                model_name_to_pred_id[name.strip().lower()] = int(idx_str)
            print(f"  Model categories: {model_name_to_pred_id}")
        else:
            return None
    else:
        return None

    remap = {}
    for name, pred_id in model_name_to_pred_id.items():
        if name in gt_name_to_id:
            gt_id = gt_name_to_id[name]
            remap[pred_id] = gt_id
            if pred_id != gt_id:
                print(f"  Remap: pred_id {pred_id} ('{name}') -> gt_id {gt_id}")

    if not remap:
        print("  WARNING: No category names matched between model and GT!")
        print("  GT names:", list(gt_name_to_id.keys()))
        print("  Model names:", list(model_name_to_pred_id.keys()))
        return None

    print(f"  Matched {len(remap)} categories "
          f"(GT has {len(gt_name_to_id)}, model has {len(model_name_to_pred_id)})")
    return remap


def remap_predictions(coco_predictions, category_remap):
    if category_remap is None:
        return coco_predictions

    remapped = []
    dropped = 0
    for pred in coco_predictions:
        pred_cat = pred["category_id"]
        if pred_cat in category_remap:
            pred_copy = pred.copy()
            pred_copy["category_id"] = category_remap[pred_cat]
            remapped.append(pred_copy)
        else:
            dropped += 1

    if dropped > 0:
        print(f"  Dropped {dropped} predictions with unmapped category IDs")
    return remapped


# ==========================================
# 3. BUILT-IN SAHI FOR LT-DETR
# ==========================================
def ltdetr_results_to_coco(results, image_id):
    """
    Convert LightlyTrain predict()/predict_sahi() results to COCO format.
    LightlyTrain outputs xyxy bboxes; COCO expects xywh.
    """
    coco_preds = []
    bboxes = results["bboxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    for bbox, score, label in zip(bboxes, scores, labels):
        xmin, ymin, xmax, ymax = bbox.tolist()
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue

        coco_preds.append({
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [xmin, ymin, w, h],
            "score": float(score),
        })
    return coco_preds


# ==========================================
# 4. UNIFIED EVALUATION LOOP
# ==========================================
def run_all_models():
    # --- User Configurations ---
    dataset_dir = "test_dataset"
    image_dir = os.path.join(dataset_dir, "images")
    gt_json_path = os.path.join(dataset_dir, "_annotations.coco.json")
    output_base_dir = "test-results-9"

    slice_height = 640
    slice_width = 640
    overlap_height_ratio = 0.1
    overlap_width_ratio = 0.1
    confidence_threshold = 0.25

    # Use LightlyTrain's built-in predict_sahi() for LT-DETR models.
    # This bypasses the custom SAHI wrapper entirely, avoiding potential issues.
    USE_BUILTIN_SAHI_FOR_LTDETR = True

    models_config = {
        "yolo26s": {
            "model_path": "runs/detect/runs/detect/yolo26s/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "yolo26l": {
            "model_path": "runs/detect/runs/detect/yolo26l3/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "rtdetr": {
            "model_path": "runs/detect/runs/detect/rtdetr-l/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "lt-tiny": {
            "model_path": "out/lt_detr_tiny/exported_models/exported_best.pt",
            "model_type": "lightly_train",
            "device": "cuda:0",
        },
        "lt-large": {
            "model_path": "out/lt_detr_large/exported_models/exported_best.pt",
            "model_type": "lightly_train",
            "device": "cuda:0",
        }
    }

    print(f"Loading Ground Truth annotations from {gt_json_path}...")
    coco_gt = COCO(gt_json_path)

    gt_img_ids = coco_gt.getImgIds()
    gt_ann_ids = coco_gt.getAnnIds()
    print(f"  GT images: {len(gt_img_ids)}, GT annotations: {len(gt_ann_ids)}")
    print(f"  GT category IDs: {sorted(coco_gt.getCatIds())}")

    sample_anns = coco_gt.loadAnns(gt_ann_ids[:3])
    for ann in sample_anns:
        cat_name = coco_gt.cats[ann['category_id']]['name']
        print(f"  Sample GT: cat_id={ann['category_id']} ('{cat_name}'), "
              f"bbox={ann['bbox']}")

    for model_name, config in models_config.items():
        print(f"\n{'='*60}\nLoading Model: {model_name}\n{'='*60}")

        is_ltdetr = config["model_type"] == "lightly_train"

        if is_ltdetr and USE_BUILTIN_SAHI_FOR_LTDETR:
            import lightly_train
            lt_model = lightly_train.load_model(config["model_path"])
            detection_model = None  # Not needed for built-in path

            # Build category remap from lt_model.classes
            gt_cats = coco_gt.cats
            gt_name_to_id = {
                info["name"].strip().lower(): cid
                for cid, info in gt_cats.items()
            }
            model_name_to_pred_id = {}
            if hasattr(lt_model, "classes") and lt_model.classes is not None:
                classes = lt_model.classes
                if isinstance(classes, dict):
                    # e.g. {0: "person", 1: "car", ...}
                    for idx, cls_name in classes.items():
                        model_name_to_pred_id[str(cls_name).strip().lower()] = int(idx)
                elif isinstance(classes, (list, tuple)):
                    # e.g. ["person", "car", ...]
                    for idx, cls_name in enumerate(classes):
                        model_name_to_pred_id[str(cls_name).strip().lower()] = idx
                else:
                    print(f"  Warning: Unexpected type for model.classes: {type(classes)}")
                    print(f"  Value: {classes}")

            print(f"  GT categories: {gt_name_to_id}")
            print(f"  Model categories: {model_name_to_pred_id}")

            category_remap = {}
            for name, pred_id in model_name_to_pred_id.items():
                if name in gt_name_to_id:
                    gt_id = gt_name_to_id[name]
                    category_remap[pred_id] = gt_id
                    if pred_id != gt_id:
                        print(f"  Remap: pred_id {pred_id} ('{name}') -> gt_id {gt_id}")

            if not category_remap:
                print("  WARNING: No category names matched!")
                category_remap = None
            else:
                print(f"  Matched {len(category_remap)} categories")

        elif is_ltdetr:
            lt_model = None
            detection_model = LightlyTrainDetectionModel(
                model_path=config["model_path"],
                confidence_threshold=confidence_threshold,
                device=config["device"]
            )
            category_remap = build_category_id_remap(
                coco_gt, detection_model, config["model_type"]
            )
        else:
            lt_model = None
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=config["model_type"],
                model_path=config["model_path"],
                confidence_threshold=confidence_threshold,
                device=config["device"],
            )
            category_remap = build_category_id_remap(
                coco_gt, detection_model, config["model_type"]
            )

        for use_sahi in [False, True]:
            mode_name = "WITH_SAHI" if use_sahi else "WITHOUT_SAHI"
            print(f"\n{'-'*40}\nRunning Mode: {mode_name}\n{'-'*40}")

            output_dir = os.path.join(output_base_dir, model_name, mode_name.lower())
            visual_samples_dir = os.path.join(output_dir, "visual_samples")
            os.makedirs(visual_samples_dir, exist_ok=True)

            print(f"Generating predictions for {model_name}...")
            coco_predictions = []

            use_builtin = is_ltdetr and USE_BUILTIN_SAHI_FOR_LTDETR and lt_model is not None

            for img_id in coco_gt.getImgIds():
                img_info = coco_gt.loadImgs(img_id)[0]
                img_file = img_info['file_name']
                img_path = os.path.join(image_dir, img_file)

                if not os.path.exists(img_path):
                    print(f"  Warning: Image not found: {img_path}")
                    continue

                if use_builtin:
                    # LightlyTrain native path
                    if use_sahi:
                        results = lt_model.predict_sahi(
                            image=img_path,
                            threshold=confidence_threshold,
                        )
                    else:
                        results = lt_model.predict(img_path)

                    img_preds = ltdetr_results_to_coco(results, image_id=img_id)
                    coco_predictions.extend(img_preds)
                else:
                    # SAHI library path
                    if use_sahi:
                        result = get_sliced_prediction(
                            img_path,
                            detection_model,
                            slice_height=slice_height,
                            slice_width=slice_width,
                            overlap_height_ratio=overlap_height_ratio,
                            overlap_width_ratio=overlap_width_ratio
                        )
                    else:
                        result = get_prediction(img_path, detection_model)

                    img_preds = result.to_coco_predictions(image_id=img_id)
                    coco_predictions.extend(img_preds)

                    result.export_visuals(
                        export_dir=visual_samples_dir,
                        file_name=os.path.splitext(img_file)[0]
                    )

            # ------------------------------------------
            # Post-process and evaluate
            # ------------------------------------------
            print(f"  Raw predictions: {len(coco_predictions)}")

            if len(coco_predictions) > 0:
                for p in coco_predictions[:3]:
                    print(f"  Sample raw pred: cat_id={p['category_id']}, "
                          f"score={p['score']:.3f}, "
                          f"bbox={[round(x, 1) for x in p['bbox']]}")

                coco_predictions = remap_predictions(coco_predictions, category_remap)
                print(f"  Remapped predictions: {len(coco_predictions)}")

                if len(coco_predictions) > 0:
                    for p in coco_predictions[:3]:
                        print(f"  Sample remapped: cat_id={p['category_id']}, "
                              f"score={p['score']:.3f}, "
                              f"bbox={[round(x, 1) for x in p['bbox']]}")

                    # Verify bbox format is xywh
                    sample = coco_predictions[0]
                    bx, by, bw, bh = sample["bbox"]
                    img_info = coco_gt.loadImgs(sample["image_id"])[0]
                    iw, ih = img_info["width"], img_info["height"]

                    if bw < 0 or bh < 0:
                        print("  ERROR: Negative w/h! Converting xyxy -> xywh...")
                        for pred in coco_predictions:
                            x1, y1, x2, y2 = pred["bbox"]
                            pred["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    elif bw > iw or bh > ih:
                        print(f"  Warning: bbox dims ({bw:.0f}x{bh:.0f}) > "
                              f"image ({iw}x{ih}). Converting xyxy -> xywh...")
                        for pred in coco_predictions:
                            x1, y1, x2, y2 = pred["bbox"]
                            pred["bbox"] = [x1, y1, x2 - x1, y2 - y1]

                    pred_json_path = os.path.join(output_dir, "predictions.json")
                    with open(pred_json_path, 'w') as f:
                        json.dump(coco_predictions, f)

                    coco_dt = coco_gt.loadRes(pred_json_path)
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()

                    print(f"\n--- OFFICIAL RESULTS: {model_name} | {mode_name} ---")
                    coco_eval.summarize()

                    stats = coco_eval.stats
                    summary = {
                        "model": model_name,
                        "mode": mode_name,
                        "AP_0.50:0.95": float(stats[0]),
                        "AP_0.50": float(stats[1]),
                        "AP_0.75": float(stats[2]),
                        "AP_small": float(stats[3]),
                        "AP_medium": float(stats[4]),
                        "AP_large": float(stats[5]),
                        "AR_1": float(stats[6]),
                        "AR_10": float(stats[7]),
                        "AR_100": float(stats[8]),
                        "AR_small": float(stats[9]),
                        "AR_medium": float(stats[10]),
                        "AR_large": float(stats[11]),
                        "num_predictions": len(coco_predictions),
                    }
                    summary_path = os.path.join(output_dir, "eval_summary.json")
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"  Summary saved to {summary_path}")
                else:
                    print("Warning: All predictions dropped after category remapping!")
            else:
                print(f"Warning: No detections by {model_name} in {mode_name} mode.")


if __name__ == "__main__":
    run_all_models()
