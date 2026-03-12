import lightly_train



if __name__ == "__main__":

    # Train

    lightly_train.train_object_detection(

        out="runs/detect/lt_detr_vits16",

        model="dinov3/vits16-ltdetr-coco",



        data={

            "path": "/fs/scratch/PAS2152/tclute/bv/final_dataset",

            "train": "images/train",

            "val": "images/val",

            "names": {

                0: "person",

                1: "tent",

            },

        },



        batch_size=16,
        num_workers=4,


        model_args={

            "lr": 0.0001,

        },



        overwrite=True,



        logger_args={

            "wandb": {

                "name": "lt_detr_vits16",

                "project": "BuckeyeVertical",

                "log_model": False,

                "offline": False,

                "anonymous": False,

                "prefix": "",

            },

        }

    )



    print("training complete")
