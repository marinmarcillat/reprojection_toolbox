import object_detection.fifty_one_utils as fou
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone import types as fot
from object_detection.sahi_inference import sahi_inference
from object_detection import inference
import object_detection.spliter as spliter
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
import numpy as np
import json
import os

def create_ocean_spy_project(image_dir, export_dir, object_detection_model_path, filtering=None, mapping = None, confidence_threshold = 0.50):
    if filtering is None:
        filtering = []
    if mapping is None:
        mapping = {}

    img_wo_annotations_dir = os.path.join(export_dir, "img_wo_annotations")
    img_w_annotations_dir = os.path.join(export_dir, "img_w_annotations")
    dataset_dir = os.path.join(export_dir, "dataset")
    for d in [img_wo_annotations_dir, img_w_annotations_dir, dataset_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    print("importing dataset")
    dataset = fou.import_image_directory(image_dir, "inference_dataset")

    print("Doing inference")
    #dataset = inference.YOLO_inference(dataset, object_detection_model_path)
    dataset = sahi_inference(object_detection_model_path, dataset)

    print("Filtering detections")
    dataset.filter_labels("detections", F("confidence") > confidence_threshold)
    dataset = dataset.filter_labels(
        "detections", F("label").is_in(filtering)
    )
    dataset = dataset.map_labels("detections", mapping)
    dataset.default_classes = fou.get_classes(dataset)
    print(f"Default classes: {dataset.default_classes}")

    print("Tiling dataset")
    dataset = spliter.dataset_tiler(dataset, img_wo_annotations_dir, 2000)

    transparency = 180
    color_palette = [
        (228, 26, 28,transparency),
        (55, 126, 184,transparency),
        (77, 175, 74,transparency),
        (152, 78, 163,transparency),
        (255, 127, 0,transparency),
        (255, 255, 51,transparency),
        (166, 86, 40,transparency),
        (247, 129, 191,transparency),
        (153, 153, 153,transparency)
    ] # Set2 palette, color-blind friendly

    color_dict = {
        label: color_palette[id]
        for id, label in enumerate(dataset.default_classes)
    }

    print("Drawing annotations")
    for sample in tqdm(dataset, total=len(dataset)):
        image = Image.open(sample.filepath)
        imr = np.array(image, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labels = sample.detections.detections

        draw = ImageDraw.Draw(image, "RGBA")

        boxes = []

        # convert bounding boxes to shapely polygons.
        for label in labels:
            l_w = label.bounding_box[2] * width
            l_h = label.bounding_box[3] * height
            x1 = label.bounding_box[0] * width
            y1 = (label.bounding_box[1]) * height
            x2 = x1 + l_w
            y2 = y1 + l_h
            draw.rectangle(((x1, y1), (x2, y2)), outline=color_dict[label["label"]], width=6)
        export = os.path.join(img_w_annotations_dir, os.path.basename(sample.filepath)).replace(".jpg", ".png")
        image.save(export)


    print("Exporting dataset")
    dataset.export(
        export_dir=dataset_dir,
        dataset_type=fot.FiftyOneImageDetectionDataset,
        label_field="detections",
    )

    color_dict_path = os.path.join(export_dir, "color_dict.json")
    with open(color_dict_path, "w") as f:
        json.dump(color_dict, f, indent=4)

    return dataset

if __name__ == '__main__':
    image_dir = r"D:\chereef_CG_2023\images"
    export_dir = r"D:\chereef_CG_2023\ocean_spy_project"
    object_detection_model_path = r"D:\model_training\trained_models\associated_species_yolov11_PC\train\weights\best.pt"
    filtering =[
        "Sabellidae",
        "SM396 Cidaris cidaris",
        "SM60 Actiniaria msp41",
        "Squat lobsters",
        "SM235 Bathynectes longispina",
        "Anemones and anemone-like",
        "Antipathidae",
        "SM631 Trochoidea  msp2"
    ]
    mapping = {
        "SM396 Cidaris cidaris": "Cidaris cidaris",
        "SM631 Trochoidea  msp2": "Trochoidea  msp2",
        "SM60 Actiniaria msp41": "Anémone msp41",
        "Squat lobsters": "Galathées",
        "SM235 Bathynectes longispina": "Bathynectes longispina",
        "Anemones and anemone-like": "Anémones",
    }

    dataset = create_ocean_spy_project(image_dir, export_dir, object_detection_model_path, filtering=filtering, mapping = mapping)

    session = fo.launch_app(dataset)
    session.show()
    session.wait()

