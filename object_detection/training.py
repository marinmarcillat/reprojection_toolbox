from object_detection.fifty_one_utils import export_yoloV5_format
from object_detection.spliter import dataset_tiler
import ultralytics.data.build as build
from object_detection.weightedDataset import YOLOWeightedDataset
from object_detection.cross_validation import k_fold_cross_validation, kfcv_training
from object_detection.hyperparameter_tuning import run_ray_tune
from ultralytics import YOLO
from pathlib import Path

def training_pipeline(dataset, model_path, project_dir, mapping=None, tiled_image_splitter = True,
                      cross_validation = True, weighted_data_loader = True, hyperparameter_tuning = False, **args):
    if mapping is None:
        mapping = {}
    ds = dataset.clone()

    project_dir = Path(project_dir)
    training_ds_export_dir = project_dir / "yolo_training_dataset"
    temp_dir = project_dir / "temp"
    training_ds_export_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)


    if mapping:
        ds = ds.map_labels("detections", mapping)

    if tiled_image_splitter:
        ds = dataset_tiler(ds, temp_dir, 2000)

    if cross_validation and not hyperparameter_tuning:
        ds_yamls = k_fold_cross_validation(ds, training_ds_export_dir)
    else:
        ds_yamls = [export_yoloV5_format(ds, training_ds_export_dir, list(ds.default_classes))]

    if weighted_data_loader:
        build.YOLODataset = YOLOWeightedDataset

    if cross_validation and not hyperparameter_tuning:
        return kfcv_training(ds_yamls, model_path, project_dir, **args)
    elif hyperparameter_tuning:
        model = YOLO(model_path, task="detect")
        return run_ray_tune(model, ds_yamls[0], tune_dir=project_dir, **args)
    else:
        model = YOLO(model_path, task="detect")
        return model.train(data=ds_yamls[0], project=project_dir, **args)



