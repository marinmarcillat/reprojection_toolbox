import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import fiftyone as fo
from tqdm import tqdm
import fifty_one_utils as fou
import os


def dataset_tiler(dataset, export_path, slice_size):
    # Adapted from https://github.com/slanj/yolo-tiling/blob/main/tile_yolo.py
    ds_name = fou.generate_rd_suffix("sliced_dataset")

    sliced_dataset = fo.Dataset(ds_name)

    new_samples = []
    for sample in tqdm(dataset, total=len(dataset)):
        im = Image.open(sample.filepath)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labels = sample.detections.detections

        boxes = []

        # convert bounding boxes to shapely polygons.
        for label in labels:
            l_w = label.bounding_box[2] * width
            l_h = label.bounding_box[3] * height
            x1 = label.bounding_box[0] * width
            y1 = (1-label.bounding_box[1]) * height
            x2 = x1 + l_w
            y2 = y1 - l_h

            boxes.append((label.label, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

        counter = 0
        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                X1 = j * slice_size
                Y1 = height - (i * slice_size)
                X2 = ((j + 1) * slice_size) - 1
                Y2 = (height - (i + 1) * slice_size) + 1

                pol = Polygon([(X1, Y1), (X2, Y1), (X2, Y2), (X1, Y2)])

                imsaved = False
                detections = []

                for box in boxes:
                    if box[1].area != 0:
                        # if the intersection area is greater than 10% of the box area, we consider it as a valid detection
                        if pol.intersects(box[1]) and pol.intersection(box[1]).area/box[1].area > 0.1:
                            inter = pol.intersection(box[1])

                            if not imsaved:
                                filename = os.path.basename(sample.filepath)
                                slice_path = os.path.join(export_path, filename.replace('.jpg', f'_{i}_{j}.jpg'))
                                if not os.path.exists(slice_path):
                                    sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                                    sliced_im = Image.fromarray(sliced)
                                    sliced_im.save(slice_path)
                                imsaved = True

                            new_box = inter.envelope

                            # get central point for the new bounding box
                            centre = new_box.centroid

                            # get coordinates of polygon vertices
                            x, y = new_box.exterior.coords.xy

                            # get bounding box width and height
                            new_width = (max(x) - min(x))
                            new_height = (max(y) - min(y))

                            # we have to normalize central x and invert y for yolo format
                            new_x = (centre.coords.xy[0][0] - new_width/2 - X1 ) / slice_size
                            new_y = (Y1 - centre.coords.xy[1][0] - new_height/2) / slice_size

                            counter += 1

                            bb = [new_x, new_y, new_width /slice_size, new_height/slice_size]

                            detections.append(fo.Detection(label=box[0], bounding_box=bb))

                if len(detections) > 0:
                    metadata = fo.ImageMetadata.build_for(slice_path)
                    new_sample = fo.Sample(filepath=slice_path, metadata=metadata, detections=fo.Detections(detections=detections))
                    new_samples.append(new_sample)
    sliced_dataset.add_samples(new_samples)
    return sliced_dataset


if __name__ == '__main__':
    dataset_dir = r"D:\model_training\trained_models\csv_corals_yolov8_PC\data"
    image_save_dir = r"D:\model_training\trained_models\csv_corals_yolov8_PC\sliced_images"
    export_dir = r"D:\model_training\trained_models\csv_corals_yolov8_PC\sliced_yolo"

    selection = ["coral"]

    dataset = fou.import_yolov5_format(dataset_dir)

    sliced_dataset = dataset_tiler(dataset, image_save_dir, 2000)

    fou.export_yoloV5_format(sliced_dataset, export_dir, selection)

