from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/yolov12-training/yolov12l/weights/last.pt")
    model.train(resume=True, epochs=140)

if __name__ == "__main__":
    main()
