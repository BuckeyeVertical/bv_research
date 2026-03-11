from pathlib import Path


DATA_DIR = "/fs/scratch/PAS2152/tclute/bv/final_dataset/"
COCO_DATA_DIR = "/fs/scratch/PAS2152/tclute/bv/coco_dataset/"
OUT_DIR = "runs/detect/"
TEST_3626_DIR = "data/flight-3-6-26"

PROJECT = "bv_research"

MODELS_CONFIG = {
        "yolo26s": {
            "model_path": "/fs/scratch/PAS2152/trev/runs/detect/runs/detect/yolo26s/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "yolo26l": {
            "model_path": "/fs/scratch/PAS2152/trev/runs/detect/runs/detect/yolo26l3/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "rtdetr": {
            "model_path": "/fs/scratch/PAS2152/trev/runs/detect/runs/detect/rtdetr-l/weights/best.pt",
            "model_type": "ultralytics",
            "device": "cuda:0",
        },
        "lt-tiny": {
            "model_path": "/fs/scratch/PAS2152/trev/out/lt_detr_tiny/exported_models/exported_best.pt",
            "model_type": "lightly_train",
            "device": "cuda:0",
        },
        "lt-large": {
            "model_path": "/fs/scratch/PAS2152/trev/out/lt_detr_large/exported_models/exported_best.pt",
            "model_type": "lightly_train",
            "device": "cuda:0",
        },
        "rf-base": {
            "model_path": "runs/detect/rf-base/checkpoint_best_ema.pth",
            "model_type": "rfdetr",
            "device": "cuda:0",
        },

}


