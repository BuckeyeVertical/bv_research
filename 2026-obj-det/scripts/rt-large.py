from ultralytics import RTDETR
import config


def main():
    # Load the pretrained RT-DETR-L model
    model = RTDETR("rtdetr-l.pt")

    # Train
    results = model.train(
        data=config.DATA_DIR,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=50,
        save=True,
        project="runs/detect",
        name="rtdetr-l",
    )

if __name__ == "__main__":
    main()
