from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='data/data.yaml',
        epochs=20,
        batch=8,
        imgsz=640,
        name='car_detection',
        val=True,
        save=True
    )

if __name__ == "__main__":
    main()
