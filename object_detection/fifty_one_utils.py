import fiftyone as fo
import fiftyone.utils.random as four
import random
import string
from PIL import Image
import os
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from fiftyone import ViewField as F
import cv2
from shutil import copy2
import object_detection.biigle_utils as biigle_utils

def make_yolo_row(label, target):
    xtl, ytl, w, h = label.bounding_box
    xc = xtl + 0.5 * w
    yc = ytl + 0.5 * h
    return "%d %f %f %f %f" % (target, xc, yc, w, h)

def get_classes(dataset, field = "detections"):
    if dataset.default_classes:
        return dataset.default_classes
    label_list = []
    for sample in dataset.head(1000):
        if sample.detections is not None:
            label_list.extend(detect.label for detect in sample[field].detections)
    return list(set(label_list))

def relative_to_absolute(bbox, w, h):
    x, y, width, height = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + width) * w)
    y2 = int((y + height) * h)
    return [x1, y1, x1, y2, x2, y2, x2, y1]

def annotation_to_biigle(dataset):
    ann_dict = {}
    for sample in dataset:
        image_path = sample.filepath

        metadata = sample.metadata
        if metadata is None:
            metadata = fo.ImageMetadata.build_for(image_path)
        w, h = metadata.width, metadata.height
        detections = sample.detections.detections

        annotations = []
        for detection in detections:
            bbox = detection.bounding_box
            bbox = relative_to_absolute(bbox, w, h)
            label = detection.label
            annotations.append({"shape_id": 5,
                               "points": bbox,
                                "label":label})
        ann_dict[image_path] = annotations
    return ann_dict

def export_to_biigle(dataset, biigle_dir, image_volume, copy = True, const_label_id = 6665):

    ann_dict = annotation_to_biigle(dataset)
    for img_path in ann_dict.keys():
        img_filename = os.path.basename(img_path)
        if not os.path.exists(os.path.join(biigle_dir, img_filename)):
            print(f"Image {os.path.join(biigle_dir, img_filename)} not found in Biigle")
            if copy:
                copy2(img_path, os.path.join(biigle_dir, img_filename))
                print(f"Copied {img_filename} to Biigle")
            else:
                print(f"Skipping {img_filename}")
                continue
        else:
            continue


        image_id = [image.image_id for image in image_volume.image if image.filename == img_filename]
        if len(image_id) == 0:
            print(f"Image {img_filename} not found in Biigle, adding it")
            if os.path.exists(os.path.join(biigle_dir, img_filename)):
                resp = image_volume.add_image([img_filename])
                image_id = resp[0]["id"]
            else:
                print(f"Image {img_filename} still not found in Biigle, skipping")
                continue
        else:
            image_id = image_id[0]

        image = [image for image in image_volume.image if image.image_id == image_id][0]
        ann_list = [biigle_utils.ImageAnnotation(0, image, ann["points"], ann["label"], const_label_id, ann["shape_id"])
                    for ann in ann_dict[img_path]]
        image.add_annotation(ann_list)
        print(f"Exported {img_filename} to Biigle")

    print("Exported all images to Biigle")

def biigle_ann_to_fiftyone(points, w, h):
    # Bounding box coordinates should be relative values
    # in [0, 1] in the following format:
    # [top-left-x, top-left-y, width, height]

    if len(points) == 3:  # circle
        x, y, r = points
        bounding_box = [(x - r) / w, (y - r) / h, (2 * r) / w, (2 * r) / h]
    else:  # polygon or rectangle
        coords = list(zip(*[iter(points)] * 2))
        min_x = min([i[0] for i in coords])
        max_x = max([i[0] for i in coords])
        min_y = min([i[1] for i in coords])
        max_y = max([i[1] for i in coords])
        bounding_box = [min_x / w, min_y / h, (max_x - min_x) / w, (max_y - min_y) / h]
    return bounding_box

def append_detections(annotations,  w, h):
    detections = []
    for ind, row in annotations.iterrows():

        bounding_box = biigle_ann_to_fiftyone(row["points"], w, h)

        detections.append(
            fo.Detection(label=row["label_name"], bounding_box=bounding_box)
        )
    return detections

def extract_frames(dataset, output_dir = None):
    output_dir = output_dir if output_dir is not None else output_dir
    stage = fo.ToFrames(sample_frames=True, sparse = True, output_dir = output_dir)
    frames_dataset = dataset.add_stage(stage)
    return frames_dataset

def import_image_csv_report(report_file, image_dir):
    report = pd.read_csv(report_file)
    report['points'] = report.points.apply(lambda x: literal_eval(str(x)))

    samples = []
    for filepath in tqdm(os.listdir(image_dir)):
        if filepath.endswith(".jpg"):
            annotations = report[report.filename == filepath]

            if len(annotations) != 0:
                metadata = fo.ImageMetadata.build_for(os.path.join(image_dir, filepath))
                sample = fo.Sample(filepath=os.path.join(image_dir, filepath), metadata=metadata)

                im = Image.open(os.path.join(image_dir, filepath), mode = "r")
                w, h = im.size

                detections = append_detections(annotations, w, h)

                # Store detections in a field name of your choice
                sample["detections"] = fo.Detections(detections=detections)

                samples.append(sample)
    return samples

def import_video_csv_report(report_file, video_dir, export_dir, w = 1920, h = 1080, frame_rate = 25):
    report = pd.read_csv(report_file)
    report['points'] = report.points.apply(lambda x: literal_eval(str(x))[0])
    report['frames'] = report.frames.apply(lambda x: literal_eval(str(x))[0])

    # Create video sample with frame labels
    samples = []
    for filepath in tqdm(os.listdir(video_dir)):
        if filepath.endswith(".mp4"):
            video_path = os.path.join(video_dir, filepath)

            annotations = report[report.video_filename == filepath]
            frames = list(set(annotations['frames'].to_list()))
            frames_dict = {x: int(round(x * frame_rate)) for x in frames}

            cap = cv2.VideoCapture(video_path)

            for frame_s, frame_nb in frames_dict.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb-1)
                ret, frame = cap.read()
                img_path = os.path.join(export_dir, f"{filepath}_{frame_nb}.jpg")
                cv2.imwrite(img_path, frame)
                metadata = fo.ImageMetadata.build_for(img_path)

                sample = fo.Sample(filepath=img_path, metadata=metadata)
                frame_annotations = annotations[annotations.frames == frame_s]

                detections = append_detections(frame_annotations, w, h)

                # Store object detections
                sample["detections"] = fo.Detections(detections=detections)

                samples.append(sample)
    return samples

def fo_annotation_to_biigle(detection: fo.Detection, img_w, img_h):
    bb = detection.bounding_box
    x1 = bb[0]*img_w
    y1 = bb[1]*img_h
    w = bb[2]*img_w
    h = bb[3]*img_h
    return [x1, y1, x1 + w, y1,x1 + w, y1 + h, x1, y1 + h]

def generate_rd_suffix(name): # Generate random name
    return name + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

def import_coco_format(dataset_dir, name):
    name = generate_rd_suffix(name)
    # Load COCO formatted dataset
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        dataset_dir=dataset_dir,
        name=name
    )
    return coco_dataset

def import_image_directory(image_dir, name):
    name = generate_rd_suffix(name)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=image_dir,
        name=name
    )
    return dataset

def map_and_filter_labels(dataset, mapping_dict, selection, label_field="detections"):
    renamed = dataset.map_labels(label_field, mapping_dict)
    filtered = renamed.filter_labels(label_field, F("label").is_in(selection))
    return filtered

def import_yolov5_format(dataset_dir, splits=None, label_field="detections"):
    if splits is None:
        splits = ["train", "val"]
    name = "yolo_v5_import"
    name = generate_rd_suffix(name)
    dataset = fo.Dataset(name)
    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=split,
            label_field=label_field,
        )
    return dataset

def export_yoloV5_format(dataset, export_dir, classes,  label_field = "detections", train_split = 0.9):
    four.random_split(
        dataset,
        {"train": train_split, "val": 1-train_split}
    )
    splits = ["train", "val"]
    for split in splits:
        split_view = dataset.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=classes,
        )
    return export_dir

def fo_to_csv(dataset):
    results = []
    for sample in dataset.iter_samples(progress=True):
        detections = sample.detections.detections
        im = Image.open(sample.filepath, mode="r")
        w, h = im.size
        for detection in detections:
            if detection.confidence > 0.3:
                points = fo_annotation_to_biigle(detection, w, h)
                results.append({
                    "filename": sample.filename,
                    "label_name": detection.label,
                    "confidence": detection.confidence,
                    "points": points,
                })
    return results

def view_dataset(dataset):
    session = fo.launch_app(dataset)
    session.wait()

