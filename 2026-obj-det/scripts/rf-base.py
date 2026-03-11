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
    # batch_size=4,
    resolution=560,
    output_dir= config.OUT_DIR + "rf-base",
    wandb=True,
    project=config.PROJECT,
    run="rf-base"
)
