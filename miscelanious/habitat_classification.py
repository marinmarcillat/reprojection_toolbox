import pandas as pd
from shutil import copy2
import os
from tqdm import tqdm
import fiftyone as fo
from object_detection.fifty_one_utils import get_classes, export_yoloV5_format

classes_labels_file = r"D:\01_canyon_scale\classification\data\classes_guide_habref.csv"
classes_names = r"D:\01_canyon_scale\classification\data\classes_names_habref.csv"

p1_labels_file = r"D:\01_canyon_scale\classification\data\333_csv_image_label_report\333-images-10m-p1.csv"
p2_labels_file = r"D:\01_canyon_scale\classification\data\335_csv_image_label_report\335-images-10m-p2.csv"

possible_image_dir = [r"Z:\images\CHEREEF_Images_Maelle\images", r"Z:\images\CHEREEF_Images_Maelle\images_P2"]

name = "habref_classification_dataset"
dataset_dir = r"D:\01_canyon_scale\classification\data\dataset_habref"

training_ds_export_dir = r"D:\01_canyon_scale\classification\data\dataset_yolo"

classes_labels = pd.read_csv(classes_labels_file)

p1_labels = pd.read_csv(p1_labels_file)
p2_labels = pd.read_csv(p2_labels_file)
all_labels = pd.concat([p1_labels, p2_labels], axis=0)
all_labels.reset_index(drop=True, inplace=True)
all_labels = all_labels[["filename", "label_name"]].merge(classes_labels, left_on='label_name', right_on='Label_name', how='left')
all_labels = all_labels[["filename", "Class"]].dropna(axis=0, how='any')

# keep only rows where Class != 0
#all_labels = all_labels[all_labels["Class"] != 0].reset_index(drop=True)

for id, row in tqdm(all_labels.iterrows()):
    for image_dir in possible_image_dir:
        image_path = os.path.join(image_dir, row['filename'])
        if os.path.exists(image_path):
            dest_dir = os.path.join(dataset_dir, str(row['Class']))
            os.makedirs(dest_dir, exist_ok=True)
            copy2(image_path, dest_dir)
            break
print("Done copying images.")

"""
ds = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
    name=name,
)

label_list = []
for sample in ds:
    if sample.ground_truth is not None:
        label_list.extend(sample.ground_truth.label)

export_yoloV5_format(ds, training_ds_export_dir, list(set(label_list)), label_field = "ground_truth")
"""




