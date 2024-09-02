from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.train(
        data="plots.yaml",
        close_mosaic=61,
        imgsz=600,
        lr0=0.03289315222021393,
        momentum=0.27808322200177155,
        batch=6,
        weight_decay=7.357608804618422e-06,
        epochs=250,
        verbose=False,
        exist_ok=True,
    )
