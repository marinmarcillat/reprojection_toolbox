from ultralytics import YOLO

def YOLO_inference(dataset, model_path):
    model = YOLO(model_path)

    dataset.apply_model(model, label_field="detections")

    return dataset