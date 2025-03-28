from object_detection.fifty_one_utils import import_image_directory
from object_detection import inference
from annotation_conversion_toolbox import biigle_dataset
import fiftyone as fo

from Biigle.biigle import Api

images_dir = r"D:\tests\active_learning"
model_path = r"D:\model_training\trained_models\associated_species_yolov11_PC\train\weights\best.pt"
export_dir = r"D:\tests\active_learning"

api = Api()
volume_id = 334
label_tree_id = 60
biigle_dir = r"Z:\images\chereef_2022\pl814_ODIS"

importer = biigle_dataset.BiigleDatasetImporter(api = api,
            label_tree_id = label_tree_id,
            volume_id = volume_id,
            volume_dir = biigle_dir)
dataset = fo.Dataset.from_importer(importer)

dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
)

session = fo.launch_app(dataset)
session.wait()



