from rfdetr import RFDETRBase
import os
import config

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12366"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

model = RFDETRBase()

model.train(
    dataset_dir=config.COCO_DATA_DIR,
    epochs=75,
    batch_size=8,
    grad_accum_steps=4,
    
    # --- STABILITY ADJUSTMENTS ---
    lr=5e-5,               # Reduced from 1e-4 to prevent gradient explosion
    lr_encoder=5e-6,       # Scaled down proportionally (backbone needs a lower LR)
    weight_decay=1e-4,     # Reduced slightly; 5e-4 can be too aggressive for fine-tuning
    amp=False,             # Disabled to prevent FP16 underflow/overflow in the matcher
    clip_max_norm=0.1,     # [Added] Crucial for DETR: prevents gradients from exploding
    # -----------------------------
    
    resolution=560,
    device="cuda",
    use_ema=True,
    early_stopping=True,
    early_stopping_patience=25,
    tensorboard=True,
    output_dir="./out/rf-base-sahi",
    wandb=True,
    project=config.PROJECT,
    run="rf-base-sahi"
)
