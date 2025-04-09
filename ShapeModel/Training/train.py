import argparse
import yaml
import torch
import wandb
import os
from ultralytics import YOLO

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train YOLO model with a given config file")
parser.add_argument("--config", required=True, help="Path to the config YAML file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Load the model
model = YOLO(config.get("model_path", "yolo11m.pt"))

# Load wandb
wandb.login(key="02749026cc907752ec1bca72657afdfaa6c28af6") # Input key on cluster

# Extract training parameters from config
exp_name = config["name"]
yaml_path = config["class_names_path"]
n_epochs = config.get("num_epochs", 30)
bs = config.get("batch_size", 4)
gpu_id = config.get("gpu_id", 0)
img_size = config.get("img_size", 1920)
wait_num = config.get("patience_time", 5)
worker_num = config.get("num_workers", 1)
optimizer_choice = config.get("optimizer", "auto")
validate = config.get("validation_bool", True)
lr0 = config.get("learning_rate_initial", 0.0003)
lrf = config.get("learning_rate_final", 0.0006)

# Train the model
if __name__ == '__main__':
    model.train(
        data=yaml_path,
        imgsz=img_size,
        pretrained=True,
        name=exp_name,
        cos_lr=True,
        lr0=lr0,
        lrf=lrf,
        epochs=n_epochs,
        batch=bs,
        device=gpu_id,
        patience=wait_num,
        val=validate,
        workers=os.cpu_count()-12,
        project = exp_name
    )
