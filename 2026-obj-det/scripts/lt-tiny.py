import lightly_train
import config

if __name__ == "__main__":
    # Train
    lightly_train.train_object_detection(
        out=config.OUT_DIR + "lt-tiny",
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
        transform_args={
            "image_size": (640, 640),
        },
        overwrite=True,
        logger_args={
            "wandb": {
                "name": "lt-tiny",
                "project": config.PROJECT,
                "log_model": False,
                "offline": False,
                "anonymous": False,
                "prefix": "",
            },
        }
    )

    print("training complete")
