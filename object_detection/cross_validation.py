from pathlib import Path
import yaml
import pandas as pd
from object_detection.fifty_one_utils import get_classes, make_yolo_row
from sklearn.model_selection import KFold
import shutil
import datetime
from tqdm import tqdm

def k_fold_cross_validation(dataset, export_dir, ksplit = 5):
    classes = get_classes(dataset)
    classes_dict = {c: i for i, c in enumerate(classes)}
    cls_idx = sorted(classes_dict.values())

    index = [sample.id for sample in dataset]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    labels_df = labels_df.fillna(0.0)

    for sample in dataset:
        for detection in sample.detections.detections:
            labels_df.loc[sample.id, classes_dict[detection.label]] += 1

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    # Loop through supported extensions and gather image files
    images = [sample.filepath for sample in dataset]

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(Path(export_dir) / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": list(classes),
                },
                ds_y,
            )

    print("Copying images and exporting labels to new directories (YoloV5)")
    for sample in tqdm(dataset):
        for split, k_split in folds_df.loc[sample.id].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(sample.filepath, img_to_path / sample.filename)

            label_path = lbl_to_path / f"{Path(sample.filename).stem}.txt"
            with open(label_path, "w") as f:
                for detection in sample.detections.detections:
                    f.write(make_yolo_row(detection, classes_dict[detection.label]) + "\n")

    return ds_yamls


