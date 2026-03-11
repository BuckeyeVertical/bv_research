from ultralytics import YOLO
def main():
    # Load pretrained YOLOv12 model
    model = YOLO("yolo12l.pt")
    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,          # GPU id (use 'cpu' if needed)
        workers=8,
        patience=20,
        save=True,
        save_period=10,
        project="yolov12-training",
        name="yolov12l",
    )


if __name__ == "__main__":
    main()