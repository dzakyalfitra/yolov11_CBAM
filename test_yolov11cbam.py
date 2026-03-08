from ultralytics import YOLO

model = YOLO("yolo11_cbam.yaml").load("yolo11n.pt")
model.info()