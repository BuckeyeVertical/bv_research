import lightly_train
import config


if __name__ == "__main__":
    # Train
    lightly_train.train_object_detection(
        out= config.OUT_DIR + "lt_detr_large",
        model="dinov3/convnext-large-ltdetr-coco",
        data={
            "path": "/fs/scratch/PAS2152/tclute/bv/final_dataset",
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "tent",
            },
        },
        batch_size=4,
        model_args={
            "lr": 0.00001,
        },
        overwrite=True,
        logger_args={
            "wandb": {
                "name": "lt_detr_large",
                "project": config.PROJECT,
                "log_model": False,
                "offline": False,
                "anonymous": False,
                "prefix": "",
            },
        }
    )

    print("training complete")
