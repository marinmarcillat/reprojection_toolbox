import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
import os

os.environ["WANDB_SILENT"] = "true"


if __name__ == '__main__':
    data_path = r"D:\model_training\trained_models\coco_multilabel_yolov11l_datarmor\sliced_yolo\dataset.yaml"
    save_dir = r"D:\model_training\trained_models\coco_multilabel_yolov11l_datarmor\sliced_yolo_training"
    model_path = r"D:\model_training\untrained_models\yolo11l.pt"
    name = "multilabel_sliced_yolov11_luisa"
    wandb_dir = os.path.join(save_dir, "wandb")

    wandb.init(
        # set the wandb project where this run will be logged
        project="Corals",
        name=name,
        dir=save_dir,
        # track hyperparameters and run metadata
        config={
            "initial_model": "yolov11l",
            "architecture": "YOLOV11",
            "dataset": "Luisa only, sliced",
            "imgsz": 1024,
            "epochs": 100,
        }
    )

    model = YOLO(model_path)

    add_wandb_callback(model, enable_train_validation_logging=False, enable_validation_logging = False)

    #result_grid = model.tune(data=data_path, epochs=50, use_ray=True, imgsz=1024, project=save_dir, name=name)
    results = model.train(data=data_path, epochs=100, imgsz=1024, project=save_dir, name=name)
    model.val()

    wandb.finish()
