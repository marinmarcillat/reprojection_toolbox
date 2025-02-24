import ultralytics.data.build as build
from ultralytics import YOLO
from object_detection.weightedDataset import YOLOWeightedDataset
from ultralytics.cfg import  TASK2METRIC
from pathlib import Path
import ray
from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler




def short_dirname(trial):
    return "trial_" + str(trial.trial_id)

import subprocess



def run_ray_tune(
    model, data, tune_dir, grace_period: int = 10, max_samples: int = 10, **train_args
):

    space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
        "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
        "box": tune.uniform(0.02, 0.2),  # box loss gain
        "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "mosaic": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
    }

    # Put the model in ray store
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        """
        Trains the YOLO model with the specified hyperparameters and additional arguments.

        Args:
            config (dict): A dictionary of hyperparameters to use for training.

        Returns:
            None
        """
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict


    # Get dataset

    space["data"] = data

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {"cpu": 1, "gpu": 1})

    # Define the ASHA scheduler for hyperparameter search
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )

    # Define the callbacks for the hyperparameter search

    # Create the Ray Tune hyperparameter search tuner
    tune_dir = Path(tune_dir)  # must be absolute dir
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples, trial_dirname_creator=short_dirname),
        run_config=RunConfig(callbacks=[], storage_path=tune_dir),
    )

    # Run the hyperparameter search
    tuner.fit()

    # Return the results of the hyperparameter search
    return tuner.get_results()



if __name__ == "__main__":
    data_path = r"D:\tests\model_unbalanced\data\2025-02-24_5-Fold_Cross-val\split_1\split_1_dataset.yaml"
    model_path = r"D:\model_training\untrained_models\yolo11l.pt"

    build.YOLODataset = YOLOWeightedDataset
    model = YOLO(model_path)

    results = run_ray_tune(model, data_path, tune_dir = r"D:\tests\model_unbalanced\tune_dir", epochs = 50, batch = 0.95, imgsz = 1024)

    print(results)