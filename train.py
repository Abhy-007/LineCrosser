from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data = "data/dataset.yaml",
        epochs = 100,
        imgsz = 224,
        batch = 16,
        project = "runs/train",
        name ="wedding_wait"
    )

if __name__ == "__main__":
    main()