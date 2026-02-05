from ultralytics import YOLO

model_path = r"D:\01_canyon_scale\classification\raw_models\yolo11x-cls.pt"

data = r"D:\01_canyon_scale\classification\data\dataset_habref"

model = YOLO(model_path)

results = model.train(data=data, batch = 8, epochs=100, imgsz=1000, patience = 10)

