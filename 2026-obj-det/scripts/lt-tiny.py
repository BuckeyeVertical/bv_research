import lightly_train
import config

if __name__ == "__main__":
    # Train
    lightly_train.train_object_detection(
        out=config.OUT_DIR / "lt_detr_tiny",
        model="dinov3/convnext-tiny-ltdetr-coco",
        data={
            "path": config.DATA_DIR,
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "tent",
            },
        },
        batch_size=2,
        model_args={
            "lr": 0.0003,
        },
        overwrite=True,
        logger_args={
            "wandb": {
                "name": "lt_detr_tiny",
                "project": config.PROJECT,
                "log_model": False,
                "offline": False,
                "anonymous": False,
                "prefix": "",
            },
        }
    )

    print("training complete")
