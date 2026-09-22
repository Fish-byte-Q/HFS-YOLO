from ultralytics import YOLO


#model = YOLO("yolov8s.pt")
model = YOLO(r'E:\result_yolov8\yolov8\wuquanzhong\xinSGD200-bs8\xinSGD200-bs8\train40\weights\best.pt')
results = model.val(data=r"E:\yolov8-two\myvisdrone2\VisDrone.yaml", batch=8, imgsz=640, device=0, workers=0, split="test", amp=True)#, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])