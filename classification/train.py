from ultralytics import YOLO

if __name__ == "__main__":
    model_path = r"D:\model_training\untrained_models\yolo11x-cls.pt"

    data_path = r"E:\ATM\yolo_training_dataset"
    project_dir = r"E:\ATM\models"

    # Load a model
    model = YOLO(model_path)  # build a new model from YAML


    # Train the model
    results = model.train(data=data_path, epochs=100, imgsz=96, project = project_dir, batch = 64, patience = 10)

