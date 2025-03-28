import fiftyone as fo
from fiftyone import ViewField as F
from Biigle.biigle import Api
import object_detection.fifty_one_utils as fou
from object_detection.biigle_utils import ImageVolume

video_volume_id = 115
lt_id = 60
project_id = 38
videos_dir = r"Z:\videos\CHEREEF_2022\Lophelia_cliff"
export_dir = r"D:\tests\video"
biigle_dir = r"Z:\images\chereef_2022\cliff_Marcos"
image_volume_id = 462

api = Api()

fou.delete_all_datasets()
dataset = fo.Dataset(name="biigle_video_annotations")

samples = fou.import_video_volume_api(api, video_volume_id, videos_dir, biigle_dir)
dataset.add_samples(samples)

label_tree = api.get(f'label-trees/{lt_id}').json()["labels"]
label_tree = {l["name"]: l["id"] for l in label_tree}
image_volume = ImageVolume(image_volume_id, api)

fou.export_to_biigle(dataset, biigle_dir, image_volume, label_tree)

image_volume.request_sam()

session = fo.launch_app(dataset)
session.show()
session.wait()