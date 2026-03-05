import lightly_train

if __name__ == "__main__":
    # Train
    lightly_train.train_object_detection(
        out="out/lt_detr_tinye",
        model="dinov3/convnext-tiny-ltdetr-coco",
        data={
            "path": "/fs/scratch/PAS2152/tclute/bv/final_dataset",
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
                # Optional display name for the run.
                "name": "lt_detr_tiny",
                # Optional project name.
                "project": "runs-detect",
                # Optional version, mainly used to resume a previous run.
                "log_model": False,
                # Optional name for uploaded checkpoints. (default: None)
                # "checkpoint_name": "checkpoint.ckpt",
                # Optional, run offline without syncing to the W&B server. (default: False)
                "offline": False,
                # Optional, configure anonymous logging. (default: False)
                "anonymous": False,
                # Optional string to put at the beginning of metric keys.
                "prefix": "",
            },
        }
    )

    print("training complete")
