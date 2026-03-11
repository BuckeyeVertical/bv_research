import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.predict import get_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import config
from datetime import datetime


# ==========================================
# 1. CUSTOM SAHI WRAPPER FOR LT-DETR
# ==========================================
class LightlyTrainDetectionModel(DetectionModel):
    def load_model(self):
        import lightly_train
        self.model = lightly_train.load_model(self.model_path)
        if self.device is not None:
            self.model = self.model.to(self.device)

        if hasattr(self.model, "classes") and self.model.classes is not None:
            classes = self.model.classes
            if isinstance(classes, dict):
                self.category_names = [classes[k] for k in sorted(classes.keys())]
                self.category_mapping = {str(k): str(v) for k, v in classes.items()}
            elif isinstance(classes, (list, tuple)):
                self.category_names = list(classes)
                self.category_mapping = {str(i): name for i, name in enumerate(classes)}
            else:
                self.category_mapping = {str(i): str(i) for i in range(1000)}
                self.category_names = [str(i) for i in range(1000)]
        else:
            self.category_mapping = {str(i): str(i) for i in range(1000)}
            self.category_names = [str(i) for i in range(1000)]
        self.num_categories = len(self.category_mapping)

    def perform_inference(self, image: np.ndarray):
        pil_img = Image.fromarray(image)
        self._original_predictions = self.model.predict(pil_img)

    def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=None, full_shape_list=None):
        original_predictions = self._original_predictions
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
            if xmin >= xmax or ymin >= ymax:
                continue
            category_id = int(label)
            category_name = self.category_mapping.get(str(category_id), str(category_id))
            object_prediction_list.append(ObjectPrediction(
                bbox=[xmin, ymin, xmax, ymax], category_id=category_id,
                score=float(score), bool_mask=None, category_name=category_name,
                shift_amount=shift_amount, full_shape=full_shape,
            ))
        return object_prediction_list


# ==========================================
# 2. CUSTOM SAHI WRAPPER FOR RF-DETR
# ==========================================
class RFDetrDetectionModel(DetectionModel):
    def load_model(self):
        from rfdetr import RFDETRBase, RFDETRLarge
        if "large" in self.model_path.lower():
            self.model = RFDETRLarge(pretrain_weights=self.model_path)
        else:
            self.model = RFDETRBase(pretrain_weights=self.model_path)

        id2label = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
            id2label = getattr(self.model.model.config, "id2label", None)
        if hasattr(self.model, "id2label"):
            id2label = self.model.id2label

        if id2label and isinstance(id2label, dict):
            self.category_mapping = {str(k): str(v) for k, v in id2label.items()}
            self.category_names = [id2label[k] for k in sorted(id2label.keys())]
        else:
            self.category_mapping = {str(i): str(i) for i in range(100)}
            self.category_names = [str(i) for i in range(100)]
        self.num_categories = len(self.category_mapping)

    def perform_inference(self, image: np.ndarray):
        pil_img = Image.fromarray(image)
        self._original_predictions = self.model.predict(pil_img, threshold=self.confidence_threshold)

    def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=None, full_shape_list=None):
        detections = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = full_shape_list[0]
        object_prediction_list = []
        bboxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id
        if bboxes is None or len(bboxes) == 0:
            return object_prediction_list

        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            if score < self.confidence_threshold:
                continue
            xmin, ymin, xmax, ymax = bbox.tolist()
            if xmin >= xmax or ymin >= ymax:
                continue
            category_id = int(class_id)
            category_name = self.category_mapping.get(str(category_id), str(category_id))
            object_prediction_list.append(ObjectPrediction(
                bbox=[xmin, ymin, xmax, ymax], category_id=category_id,
                score=float(score), bool_mask=None, category_name=category_name,
                shift_amount=shift_amount, full_shape=full_shape,
            ))
        return object_prediction_list


# ==========================================
# 3. CATEGORY ID MAPPING HELPER
# ==========================================
def build_category_id_remap(coco_gt, detection_model, model_type):
    gt_cats = coco_gt.cats
    gt_name_to_id = {info["name"].strip().lower(): cid for cid, info in gt_cats.items()}
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
    elif model_type in ("lightly_train", "rfdetr"):
        if hasattr(detection_model, "category_mapping") and detection_model.category_mapping:
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
        print("  WARNING: No category names matched!")
        return None
    print(f"  Matched {len(remap)} categories")
    return remap


def remap_predictions(coco_predictions, category_remap):
    if category_remap is None:
        return coco_predictions
    remapped, dropped = [], 0
    for pred in coco_predictions:
        if pred["category_id"] in category_remap:
            pred_copy = pred.copy()
            pred_copy["category_id"] = category_remap[pred["category_id"]]
            remapped.append(pred_copy)
        else:
            dropped += 1
    if dropped > 0:
        print(f"  Dropped {dropped} predictions with unmapped category IDs")
    return remapped


# ==========================================
# 4. RESULT CONVERSION HELPERS
# ==========================================
def ltdetr_results_to_coco(results, image_id):
    coco_preds = []
    bboxes = results["bboxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    for bbox, score, label in zip(bboxes, scores, labels):
        xmin, ymin, xmax, ymax = bbox.tolist()
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            continue
        coco_preds.append({"image_id": image_id, "category_id": int(label), "bbox": [xmin, ymin, w, h], "score": float(score)})
    return coco_preds


def rfdetr_results_to_coco(detections, image_id):
    coco_preds = []
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return coco_preds
    for bbox, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        xmin, ymin, xmax, ymax = bbox.tolist()
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            continue
        coco_preds.append({"image_id": image_id, "category_id": int(class_id), "bbox": [xmin, ymin, w, h], "score": float(score)})
    return coco_preds


# ==========================================
# 5. VISUALIZATION HELPERS
# ==========================================
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


def draw_detections_on_image(image_path, preds_xyxy, save_path, class_names=None, score_threshold=0.0):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for det in preds_xyxy:
        score = det.get("score", 1.0)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cat_id = det.get("category_id", 0)
        color = _COLORS[cat_id % len(_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        cat_name = det.get("category_name", "")
        if not cat_name and class_names:
            cat_name = class_names.get(cat_id, str(cat_id))
        label = f"{cat_name} {score:.2f}" if cat_name else f"{cat_id} {score:.2f}"
        draw.text((x1, max(y1 - 16, 0)), label, fill=color, font=font)
    img.save(save_path)


def ltdetr_preds_to_vis_format(results):
    vis = []
    bboxes = results["bboxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    for bbox, score, label in zip(bboxes, scores, labels):
        vis.append({"bbox_xyxy": bbox.tolist(), "score": float(score), "category_id": int(label)})
    return vis


def rfdetr_preds_to_vis_format(detections):
    vis = []
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return vis
    for bbox, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        vis.append({"bbox_xyxy": bbox.tolist(), "score": float(score), "category_id": int(class_id)})
    return vis


# ==========================================
# 6. IMAGE DISCOVERY FOR LABEL-FREE MODE
# ==========================================
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def discover_images(image_dir):
    return [f for f in sorted(os.listdir(image_dir)) if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS]


# ==========================================
# 7. HELPER: BUILD REMAP FOR BUILT-IN MODELS
# ==========================================
def build_builtin_remap(coco_gt, model_name_to_pred_id):
    """Shared remap logic for LT-DETR / RF-DETR built-in paths."""
    gt_name_to_id = {info["name"].strip().lower(): cid for cid, info in coco_gt.cats.items()}
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
        return None
    print(f"  Matched {len(category_remap)} categories")
    return category_remap


def extract_class_info(classes_obj):
    """Extract model_name_to_pred_id and class_names_for_vis from a classes object."""
    model_name_to_pred_id = {}
    class_names_for_vis = {}
    if classes_obj is None:
        return model_name_to_pred_id, class_names_for_vis
    if isinstance(classes_obj, dict):
        for idx, cls_name in classes_obj.items():
            model_name_to_pred_id[str(cls_name).strip().lower()] = int(idx)
            class_names_for_vis[int(idx)] = str(cls_name)
    elif isinstance(classes_obj, (list, tuple)):
        for idx, cls_name in enumerate(classes_obj):
            model_name_to_pred_id[str(cls_name).strip().lower()] = idx
            class_names_for_vis[idx] = str(cls_name)
    return model_name_to_pred_id, class_names_for_vis


# ==========================================
# 8. UNIFIED EVALUATION LOOP
# ==========================================
def run_all_models():
    # ------------------------------------------------------------------
    # User Configurations
    # ------------------------------------------------------------------
    dataset_dir = config.TEST_3626_DIR
    image_dir = os.path.join(dataset_dir, "images")
    gt_json_path = os.path.join(dataset_dir, "_annotations.coco.json")
    output_base_dir = f"test-results/{datetime.now()}"
    confidence_threshold = 0.25

    USE_BUILTIN_FOR_LTDETR = True
    USE_BUILTIN_FOR_RFDETR = True

    # ------------------------------------------------------------------
    # Label-free mode detection
    # ------------------------------------------------------------------
    has_labels = os.path.isfile(gt_json_path)
    if has_labels:
        print(f"Loading Ground Truth annotations from {gt_json_path}...")
        coco_gt = COCO(gt_json_path)
        print(f"  GT images: {len(coco_gt.getImgIds())}, GT annotations: {len(coco_gt.getAnnIds())}")
        print(f"  GT category IDs: {sorted(coco_gt.getCatIds())}")
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds()[:3]):
            cat_name = coco_gt.cats[ann["category_id"]]["name"]
            print(f"  Sample GT: cat_id={ann['category_id']} ('{cat_name}'), bbox={ann['bbox']}")
    else:
        coco_gt = None
        print(f"No annotation file at {gt_json_path}. Running in LABEL-FREE mode.")
        image_files = discover_images(image_dir)
        if not image_files:
            print(f"ERROR: No images found in {image_dir}")
            return
        print(f"  Found {len(image_files)} images")

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------
    models_config = config.MODELS_CONFIG

    # ------------------------------------------------------------------
    # Iterate over every model
    # ------------------------------------------------------------------
    for model_name, cfg in models_config.items():
        print(f"\n{'='*60}\nLoading Model: {model_name}\n{'='*60}")

        model_type = cfg["model_type"]
        is_ltdetr = model_type == "lightly_train"
        is_rfdetr = model_type == "rfdetr"

        lt_model = None
        rf_model = None
        detection_model = None
        category_remap = None
        class_names_for_vis = {}

        # ---- LT-DETR (built-in) ----
        if is_ltdetr and USE_BUILTIN_FOR_LTDETR:
            import lightly_train
            lt_model = lightly_train.load_model(cfg["model_path"])
            classes_obj = getattr(lt_model, "classes", None)
            model_name_to_pred_id, class_names_for_vis = extract_class_info(classes_obj)
            if has_labels:
                category_remap = build_builtin_remap(coco_gt, model_name_to_pred_id)

        # ---- LT-DETR (SAHI wrapper fallback) ----
        elif is_ltdetr:
            detection_model = LightlyTrainDetectionModel(
                model_path=cfg["model_path"],
                confidence_threshold=confidence_threshold,
                device=cfg["device"],
            )
            if hasattr(detection_model, "category_mapping"):
                for idx_str, name in detection_model.category_mapping.items():
                    class_names_for_vis[int(idx_str)] = name
            if has_labels:
                category_remap = build_category_id_remap(coco_gt, detection_model, model_type)

        # ---- RF-DETR (built-in) ----
        elif is_rfdetr and USE_BUILTIN_FOR_RFDETR:
            from rfdetr import RFDETRBase, RFDETRLarge
            if "large" in cfg["model_path"].lower():
                rf_model = RFDETRLarge(pretrain_weights=cfg["model_path"])
            else:
                rf_model = RFDETRBase(pretrain_weights=cfg["model_path"])

            id2label = None
            if hasattr(rf_model, "model") and hasattr(rf_model.model, "config"):
                id2label = getattr(rf_model.model.config, "id2label", None)
            if hasattr(rf_model, "id2label"):
                id2label = rf_model.id2label

            model_name_to_pred_id, class_names_for_vis = extract_class_info(id2label)
            if has_labels:
                category_remap = build_builtin_remap(coco_gt, model_name_to_pred_id)

        # ---- RF-DETR (SAHI wrapper fallback) ----
        elif is_rfdetr:
            detection_model = RFDetrDetectionModel(
                model_path=cfg["model_path"],
                confidence_threshold=confidence_threshold,
                device=cfg["device"],
            )
            if hasattr(detection_model, "category_mapping"):
                for idx_str, name in detection_model.category_mapping.items():
                    class_names_for_vis[int(idx_str)] = name
            if has_labels:
                category_remap = build_category_id_remap(coco_gt, detection_model, model_type)

        # ---- Ultralytics (YOLO / RT-DETR) ----
        else:
            from sahi import AutoDetectionModel
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=cfg["model_path"],
                confidence_threshold=confidence_threshold,
                device=cfg["device"],
            )
            try:
                names = detection_model.model.model.names
                if isinstance(names, dict):
                    class_names_for_vis = {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    class_names_for_vis = {i: v for i, v in enumerate(names)}
            except Exception:
                pass
            if has_labels:
                category_remap = build_category_id_remap(coco_gt, detection_model, model_type)

        # ==============================================================
        # Single-shot inference on all images
        # ==============================================================
        output_dir = os.path.join(output_base_dir, model_name)
        visual_samples_dir = os.path.join(output_dir, "visual_samples")
        os.makedirs(visual_samples_dir, exist_ok=True)

        print(f"Running inference for {model_name}...")
        coco_predictions = []

        if has_labels:
            iter_items = [(img_id, coco_gt.loadImgs(img_id)[0]["file_name"]) for img_id in coco_gt.getImgIds()]
        else:
            iter_items = list(enumerate(image_files))

        use_builtin_lt = is_ltdetr and USE_BUILTIN_FOR_LTDETR and lt_model is not None
        use_builtin_rf = is_rfdetr and USE_BUILTIN_FOR_RFDETR and rf_model is not None

        for img_id, img_file in iter_items:
            img_path = os.path.join(image_dir, img_file)
            if not os.path.exists(img_path):
                print(f"  Warning: Image not found: {img_path}")
                continue

            vis_save_path = os.path.join(visual_samples_dir, os.path.splitext(img_file)[0] + ".jpg")

            if use_builtin_lt:
                results = lt_model.predict(img_path)
                coco_predictions.extend(ltdetr_results_to_coco(results, image_id=img_id))
                draw_detections_on_image(
                    img_path, ltdetr_preds_to_vis_format(results), vis_save_path,
                    class_names=class_names_for_vis, score_threshold=confidence_threshold,
                )

            elif use_builtin_rf:
                detections = rf_model.predict(img_path, threshold=confidence_threshold)
                coco_predictions.extend(rfdetr_results_to_coco(detections, image_id=img_id))
                draw_detections_on_image(
                    img_path, rfdetr_preds_to_vis_format(detections), vis_save_path,
                    class_names=class_names_for_vis, score_threshold=confidence_threshold,
                )

            else:
                result = get_prediction(img_path, detection_model)
                coco_predictions.extend(result.to_coco_predictions(image_id=img_id))
                result.export_visuals(export_dir=visual_samples_dir, file_name=os.path.splitext(img_file)[0])

        # ----------------------------------------------------------
        # Post-process and (optionally) evaluate
        # ----------------------------------------------------------
        print(f"  Raw predictions: {len(coco_predictions)}")

        if len(coco_predictions) == 0:
            print(f"Warning: No detections by {model_name}.")
            continue

        for p in coco_predictions[:3]:
            print(f"  Sample raw pred: cat_id={p['category_id']}, score={p['score']:.3f}, bbox={[round(x, 1) for x in p['bbox']]}")

        raw_pred_path = os.path.join(output_dir, "predictions_raw.json")
        with open(raw_pred_path, "w") as f:
            json.dump(coco_predictions, f)
        print(f"  Raw predictions saved to {raw_pred_path}")

        if not has_labels:
            print("  (Label-free mode: skipping COCO evaluation)")
            continue

        # --- Remap + evaluate ---
        coco_predictions = remap_predictions(coco_predictions, category_remap)
        print(f"  Remapped predictions: {len(coco_predictions)}")

        if len(coco_predictions) == 0:
            print("Warning: All predictions dropped after category remapping!")
            continue

        for p in coco_predictions[:3]:
            print(f"  Sample remapped: cat_id={p['category_id']}, score={p['score']:.3f}, bbox={[round(x, 1) for x in p['bbox']]}")

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
            print(f"  Warning: bbox dims ({bw:.0f}x{bh:.0f}) > image ({iw}x{ih}). Converting xyxy -> xywh...")
            for pred in coco_predictions:
                x1, y1, x2, y2 = pred["bbox"]
                pred["bbox"] = [x1, y1, x2 - x1, y2 - y1]

        pred_json_path = os.path.join(output_dir, "predictions.json")
        with open(pred_json_path, "w") as f:
            json.dump(coco_predictions, f)

        coco_dt = coco_gt.loadRes(pred_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        print(f"\n--- OFFICIAL RESULTS: {model_name} ---")
        coco_eval.summarize()

        stats = coco_eval.stats
        summary = {
            "model": model_name,
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
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    run_all_models()
