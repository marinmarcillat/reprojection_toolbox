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

def create_ocean_spy_project(image_dir, export_dir, object_detection_model_path, confidence_threshold = 0.50):
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

    print("Tiling dataset")
    dataset = spliter.dataset_tiler(dataset, img_wo_annotations_dir, 2000)

    color_palette = [(27,158,119,180),
                     (217,95,2,180),
                     (117,112,179,180),
                     (231,41,138,180),
                     (102,166,30,180),
                     (230,171,2,180),
                     (158,114,32,180),
                     (102,102,102,180)]  # Dark2 palette, color-blind friendly
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
    image_dir = r"F:\marin\chereef_CG_2023\images"
    export_dir = r"F:\marin\chereef_CG_2023\ocean_spy_project"
    object_detection_model_path = r"D:\model_training\trained_models\coco_multilabel_yolov11l_datarmor\train_yolo11l_100e_2000imgsz_datarmor\weights\best.pt"

    dataset = create_ocean_spy_project(image_dir, export_dir, object_detection_model_path)

    session = fo.launch_app(dataset)
    session.show()
    session.wait()

