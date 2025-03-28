from object_detection.fifty_one_utils import export_yoloV5_format, get_classes, import_image_csv_report, import_image_directory, delete_all_datasets
from object_detection.spliter import dataset_tiler
import ultralytics.data.build as build
from object_detection.weightedDataset import YOLOWeightedDataset
from object_detection.cross_validation import k_fold_cross_validation, kfcv_training
from object_detection.hyperparameter_tuning import run_ray_tune
from ultralytics import YOLO
from pathlib import Path
from fiftyone import ViewField as F

def training_pipeline(dataset, model_path, project_dir, project_name = "project", mapping=None, filtering = None, tiled_image_splitter = True,
                      cross_validation = True, weighted_data_loader = True, hyperparameter_tuning = False, training_config = None, training = True):
    if mapping is None:
        mapping = {}
    if training_config is None:
        training_config = {}
    if filtering is None:
        filtering = []

    ds = dataset.clone()

    project_dir = Path(project_dir)
    training_ds_export_dir = project_dir / f"yolo_training_dataset_{project_name}"
    temp_dir = project_dir / "temp"
    training_ds_export_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    training_config["project"] = project_dir
    training_config["name"] = project_name

    if filtering:
        ds = ds.filter_labels(
            "detections", F("label").is_in(filtering)
        )
        ds.save()

    if mapping:
        ds = ds.map_labels("detections", mapping)
        ds.save()

    ds.default_classes = get_classes(ds)
    ds.save()

    if tiled_image_splitter:
        ds = dataset_tiler(ds, temp_dir, 2000)

    if cross_validation and not hyperparameter_tuning:
        ds_yamls = k_fold_cross_validation(ds, training_ds_export_dir)
    else:
        ds_yamls = [export_yoloV5_format(ds, training_ds_export_dir, list(ds.default_classes))]

    if not training:
        return ds

    if weighted_data_loader:
        build.YOLODataset = YOLOWeightedDataset

    if cross_validation and not hyperparameter_tuning:
        return kfcv_training(ds_yamls, model_path, **training_config)
    elif hyperparameter_tuning:
        model = YOLO(model_path, task="detect")
        return run_ray_tune(model, ds_yamls[0], tune_dir=project_dir, **training_config)
    else:
        model = YOLO(model_path, task="detect")
        return model.train(data=ds_yamls[0], **training_config)



if __name__ == "__main__":

    report_path = r"D:\model_training\trained_models\associated_species_yolov11_PC\dataset_luisa_vol334_612img\334-deep-learning-coral-garden-pl814-odis_270225.csv"
    image_dir = r"Z:\images\chereef_2022\pl814_ODIS"
    dataset_name = "coral_garden_anemones"

    scenarios = [
        {
            "training": False,

            "weighted_data_loader": True,
            "tiled_image_splitter": True,
            "cross_validation": False,
            "hyperparameter_tuning": False,

            "project_dir": r"D:\model_training\trained_models\associated_species_yolov11_PC\dataset_luisa_vol334_612img",
            "model_path": r"D:\model_training\untrained_models\yolo11l.pt",
            "project_name": "latest",
            "filtering": ["Sabellidae",
                          "SM235 Bathynectes longispina",
                          "Squat lobsters",
                          "Anemones and anemone-like",
                          "SM56 Halcampoides msp1",
                          "SM92 Actiniaria msp1",
                          "SM60 Actiniaria msp41",
                          "Antipathidae",
                          "SM130 Stichopathes cf. gravieri",
                          "SM144 Antipathes cf. dichotoma",
                          "Stichopathes sp. (undefined)",
                          "SM396 Cidaris cidaris",
                          "SM631 Trochoidea  msp2",
                          "Crust-like"],
            "mapping": {
                "SM56 Halcampoides msp1": "Anemones and anemone-like",
                "SM92 Actiniaria msp1": "Anemones and anemone-like",
                "SM130 Stichopathes cf. gravieri": "Antipathidae",
                "SM144 Antipathes cf. dichotoma": "Antipathidae",
                "Stichopathes sp. (undefined)": "Antipathidae",
            },
            "training_config": {
                "imgsz": 1024,
                "batch": 0.99,
                "epochs": 100,
                "patience": 20,
            }
        }
    ]

    delete_all_datasets()

    samples = import_image_csv_report(report_path, image_dir)
    dataset = import_image_directory(image_dir, dataset_name)
    dataset.add_samples(samples)
    dataset.default_classes = get_classes(dataset)
    dataset = dataset.match(F("detections.detections").length() != 0)

    results = []
    results.extend(
        training_pipeline(dataset, **scenario) for scenario in scenarios
    )
    print(results)
