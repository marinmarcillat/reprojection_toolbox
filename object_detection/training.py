from object_detection.fifty_one_utils import import_image_csv_report, import_image_directory, get_classes
import fiftyone as fo
from object_detection.spliter import dataset_tiler
import ultralytics.data.build as build
from object_detection.weightedDataset import YOLOWeightedDataset
import pandas as pd
from object_detection.fifty_one_utils import get_classes
from object_detection.cross_validation import k_fold_cross_validation
from fiftyone import ViewField as F
from sklearn.model_selection import KFold
import datetime
import yaml
import shutil
from pathlib import Path

build.YOLODataset = YOLOWeightedDataset

report_path = r"D:\tests\model_unbalanced\334-deep-learning-coral-garden-pl814-odis.csv"
image_dir = r"Z:\images\chereef_2022\pl814_ODIS"
export_dir = r"D:\tests\model_unbalanced\data"
temp_dir = r"D:\tests\model_unbalanced\temp"


samples = import_image_csv_report(report_path, image_dir)
dataset = import_image_directory(image_dir, "test_unbalanced")
dataset.add_samples(samples)

dataset.default_classes = get_classes(dataset)

 # replace with 'path/to/dataset' for your custom data
dataset = dataset.match(F("detections.detections").length() != 0)

tiled_dataset = dataset_tiler(dataset, temp_dir, 2000)

k_fold_cross_validation(tiled_dataset, export_dir)

print("stop")


