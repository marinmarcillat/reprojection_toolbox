from object_detection.fifty_one_utils import import_image_csv_report, import_image_directory, get_classes
from object_detection.spliter import dataset_tiler
import ultralytics.data.build as build
from object_detection.weightedDataset import YOLOWeightedDataset
from object_detection.fifty_one_utils import get_classes
from object_detection.cross_validation import k_fold_cross_validation
from fiftyone import ViewField as F
from ultralytics import YOLO

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

ds_yamls = k_fold_cross_validation(tiled_dataset, export_dir)

weights_path = "path/to/weights.pt"
model = YOLO(weights_path, task="detect")

results = {}

# Define your additional arguments here
batch = 16
project = "kfold_demo"
epochs = 100

for k in range(5):
    dataset_yaml = ds_yamls[k]
    model = YOLO(weights_path, task="detect")
    model.train(data=dataset_yaml, epochs=epochs, batch=batch, project=project)  # include any train arguments
    results[k] = model.metrics  # save output metrics for further analysis

print("stop")


