from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data = "C:/Users/anshi/Desktop/garbage sorting/data/yolodata/data.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 16
)

