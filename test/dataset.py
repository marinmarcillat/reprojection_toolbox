from Biigle.biigle import Api
import os
from shutil import copy2
from ultralytics import YOLO
import fiftyone as fo
from tqdm import tqdm
from object_detection import fifty_one_utils as fou
from object_detection import sahi_inference as sahi
from annotation_conversion_toolbox.biigle_dataset import BiigleDatasetExporter

api = Api()
volume_id = 684
label_tree_id = 60
label_id = 9813
img_dir_biigle = r"Z:\images\chereef_2022\CG_disjoint"
img_biigle_dict = api.get(f"volumes/{volume_id}/filenames").json()
img_biigle_dict = {img_biigle_dict[i]: i for i in img_biigle_dict}
img_select_dir = r"D:\03-Change_detection\colonies\images_selection"

model_path = r"D:\02_object_detection\trained_models\csv_corals_yolov8_PC\training_sliced\weights\best.pt"
model = YOLO(model_path)
img_sz = model.model.args['imgsz']


for img, img_id in tqdm(img_biigle_dict.items()):

    annotations = api.get(f"images/{img_id}/annotations").json()
    if len(annotations) == 0:
        copy2(os.path.join(img_dir_biigle, img), img_select_dir)

dataset = fou.import_image_directory(img_select_dir, "Coral_CG22_0903")
dataset.compute_metadata()
first_img_size = [dataset.first()["metadata"]["width"], dataset.first()["metadata"]["height"]]

dataset_inf = sahi.sahi_inference(model_path, dataset, slice=img_sz)

view = dataset_inf.map_labels("detections", {"coral": "SM1133 Madrepora oculata"})

exporter = BiigleDatasetExporter(api=api, volume_id=volume_id, label_tree_id=label_tree_id, biigle_image_dir=img_dir_biigle)
dataset_inf.export(dataset_exporter=exporter)

print("stop")
print("end")



