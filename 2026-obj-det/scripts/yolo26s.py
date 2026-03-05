from ultralytics import YOLO


def main():
    # Load the pretrained YOLO26l model
    model = YOLO("yolo26s.pt")

    # Train
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=20,
        save=True,
        save_period=10,
        project="runs/detect",
        name="yolo26s",
    )

if __name__ == "__main__":
    main()
