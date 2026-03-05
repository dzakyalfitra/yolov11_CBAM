from ultralytics import YOLO

model = YOLO("yolo11_cbam.yaml")
model.info()