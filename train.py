
from ultralytics import YOLO

model = YOLO(r"/root/autodl-tmp/yolov11-main/ultralytics/cfg/models/11/yolo11s-NewConvBlock-NewAttention-C3k2NB-layer.yaml")
model.load("yolo11s.pt")

results = model.train(data=r"/root/autodl-tmp/yolov11-main/myvisdrone2/VisDrone.yaml", imgsz=640,epochs=200, batch=8, device=0, optimizer="SGD", workers=12,iou=0.5)
metrics=model.val()