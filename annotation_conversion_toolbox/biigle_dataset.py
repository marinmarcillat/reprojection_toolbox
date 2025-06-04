import fiftyone.utils.data as foud
import fiftyone.types as fot
import eta.core.serial as etas
import fiftyone.core.labels as fol

import os
from shutil import copy2
from PIL import Image

def bb_relative_to_absolute(bbox, w, h):
    x, y, width, height = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + width) * w)
    y2 = int((y + height) * h)
    return [x1, y1, x1, y2, x2, y2, x2, y1]

def pl_relative_to_absolute(points, w, h, closed = True):
    coords = []
    for x, y in points:
        coords.extend((x * w, y * h))
    if closed:
        coords.extend(coords[:2])
    return coords


def absolute_to_relative(point, w, h):
    x, y = point
    x = x / w
    y = y / h
    return x, y

def biigle_ann_to_fiftyone(label, points, w, h, shape_id, conf = 1.0):
    # Bounding box coordinates should be relative values
    # in [0, 1] in the following format:
    # [top-left-x, top-left-y, width, height]

    if shape_id == 4:  # circle, BB
        x, y, r = points
        bounding_box =  [(x - r) / w, (y - r) / h, (2 * r) / w, (2 * r) / h]
        return fol.Detection(
            label=label, bounding_box=bounding_box, confidence=conf
        )

    elif shape_id == 5: #rectangle, BB
        coords = list(zip(*[iter(points)] * 2))
        min_x = min(i[0] for i in coords)
        max_x = max(i[0] for i in coords)
        min_y = min(i[1] for i in coords)
        max_y = max(i[1] for i in coords)
        bounding_box = [min_x / w, min_y / h, (max_x - min_x) / w, (max_y - min_y) / h]

        return fol.Detection(
            label=label, bounding_box=bounding_box, confidence=conf
        )

    elif shape_id in [2, 3]:  #  polyline or polygon
        
        coords = list(zip(*[iter(points)] * 2))
        points = [absolute_to_relative(point, w, h) for point in coords]
        return fol.Polyline(
            label=label,
            points=[points],
            confidence=conf,
            closed=(shape_id == 3),
            filled=False,
        )
    elif shape_id == 7:  # classification
        return fol.Classification(label=label, confidence=conf)

class BiigleDatasetExporter(foud.LabeledImageDatasetExporter, foud.ExportPathsMixin):

    def __init__(
        self,
        api,
        volume_id,
        label_tree_id,
        biigle_image_dir,
        rel_dir=None,
        abs_paths=False,
        classes=None,
        include_confidence=None,
        include_attributes=None,
        image_format=None,
        pretty_print=True,
    ):


        super().__init__()

        self.api = api

        self.volume_id = volume_id
        self.label_tree_id = label_tree_id
        self.biigle_image_dir = biigle_image_dir

        self.rel_dir = rel_dir
        self.abs_paths = abs_paths
        self.classes = classes
        self.include_confidence = include_confidence
        self.include_attributes = include_attributes
        self.image_format = image_format
        self.pretty_print = pretty_print

        self._labels_map_rev = None
        self._labels_dict = None
        self._volume_img_dict = None
        self._annotations = []
        self._img_to_add = []

    @property
    def requires_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return fol.Detections, fol.Polylines, fol.Classifications

    def setup(self):
        label_tree = self.api.get(f'label-trees/{self.label_tree_id}').json()["labels"]
        self._labels_dict = {l["name"]: l["id"] for l in label_tree}

        img_dict = self.api.get(f'volumes/{self.volume_id}/filenames').json()
        self._volume_img_dict = {x: y for y, x in img_dict.items()}

    def export_sample(self, image_path, label, metadata):
        image_filename = os.path.basename(image_path)
        target_dest = os.path.join(self.biigle_image_dir, image_filename)
        if not os.path.exists(target_dest):
            copy2(image_path, target_dest)
        w, h = metadata.width, metadata.height

        if image_filename not in self._volume_img_dict:
            self._img_to_add.append(image_filename)

        annotation_list = []
        if isinstance(label, fol.Polylines):
            polylines = label.polylines
            for polyline in polylines:
                coordinates = pl_relative_to_absolute(polyline.points, w, h, polyline.closed)
                shape_id = 3 if polyline.closed else 2
                label = polyline.label
                label_id = self._labels_dict[label]
                confidence = polyline.confidence
                annotation = {"image": image_filename,"shape_id":shape_id, "label_id":label_id, "confidence":confidence, "points":coordinates}
                annotation_list.append(annotation)

        elif isinstance(label, fol.Detections):
            detections = label.detections

            shape_id = 5
            for detection in detections:
                coordinates = bb_relative_to_absolute(detection.bounding_box, w, h)
                label_id = self._labels_dict[detection.label]
                confidence = detection.confidence
                annotation = {"image": image_filename,"shape_id": shape_id, "label_id": label_id, "confidence": confidence,
                              "points": coordinates}
                annotation_list.append(annotation)

        elif isinstance(label, fol.Classifications):
            classifications = label.classifications
            shape_id = 7
            for classification in classifications:
                label = classification.label
                label_id = self._labels_dict[label]
                confidence = classification.confidence
                annotation = {"image": image_filename,"shape_id": shape_id, "label_id": label_id, "confidence": confidence}
                annotation_list.append(annotation)

        self._annotations.extend(annotation_list)


    def close(self, *args):
        if len(self._img_to_add) != 0:
            self.api.post(f'volumes/{self.volume_id}/files', json={"files": self._img_to_add})

        img_dict = self.api.get(f'volumes/{self.volume_id}/filenames').json()
        self._volume_img_dict = {x: y for y, x in img_dict.items()}

        for annotation in self._annotations:
            image_id = self._volume_img_dict[annotation["image"]]
            del annotation["image"]
            annotation["image_id"] = int(image_id)

        chunks =  [self._annotations[i:i + 99] for i in range(0, len(self._annotations), 99)]

        for chunk in chunks:
            self.api.post('image-annotations', json=chunk)


    def _parse_classes(self):
        if self.classes is not None:
            self._labels_map_rev = foud._to_labels_map_rev(self.classes)


class BiigleDatasetImporter(foud.LabeledImageDatasetImporter, foud.ImportPathsMixin):
    """Custom importer for labeled image datasets.

    Args:
        dataset_dir (None): the dataset directory. This may be optional for
            some importers
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
        **kwargs: additional keyword arguments for your importer
    """

    def __init__(
            self,
            api,
            label_tree_id,
            volume_id,
            volume_dir,
            shuffle=False,
            seed=None,
            max_samples=None,
            **kwargs,
    ):

        super().__init__(
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.api = api
        self.label_tree_id = label_tree_id
        self.volume_id = volume_id
        self.volume_dir = volume_dir

        self._info = None
        self._classes = None
        self._labels_paths_map = None
        self._labels_map = None
        self._file_ids = None
        self._iter_file_ids = None
        self._num_samples = None
        self._volume_img_dict = None

    def __iter__(self):
        self._iter_file_ids = iter(self._file_ids)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        file_id = next(self._iter_file_ids)
        filepath = os.path.join(self.volume_dir, self._volume_img_dict[file_id])

        im = Image.open(os.path.join(self.volume_dir, filepath), mode="r")
        w, h = im.size

        annotations = self.api.get(f'images/{file_id}/annotations').json()
        label_dict = None
        if annotations:
            label_dict = {}
            detections = []
            polylines = []
            classifications = []

            for annotation in annotations:
                shape = annotation["shape_id"]
                label = annotation["labels"][0]["label"]["name"]
                lab = biigle_ann_to_fiftyone(label, annotation["points"], w, h, shape)

                if shape in [5, 4]:
                    detections.append(lab)
                elif shape in [2, 3]:
                    polylines.append(lab)
                elif shape in [7]:
                    classifications.append(lab)
            if detections:
                label_dict["detections"] = fol.Detections(detections=detections)
            if polylines:
                label_dict["polylines"] = fol.Polylines(polylines=polylines)
            if classifications:
                label_dict["classifications"] = fol.Classifications(classifications=classifications)

        return filepath, None, label_dict

    @property
    def has_dataset_info(self):
        return True

    @property
    def has_image_metadata(self):
        return False

    @property
    def label_cls(self):
        return {"detections": fol.Detections, "polylines": fol.Polylines, "classifications": fol.Classifications}

    def setup(self):
        label_tree = self.api.get(f'label-trees/{self.label_tree_id}').json()["labels"]
        self._labels_dict = {l["name"]: l["id"] for l in label_tree}

        self._volume_img_dict = self.api.get(f'volumes/{self.volume_id}/filenames').json()
        self._file_ids = self._volume_img_dict.keys()

        # set the setup here

    def get_dataset_info(self):
        return self._info

    def close(self, *args):
        pass


class BiigleImageDataset(fot.LabeledImageDataset):
    """Custom unlabeled image dataset type."""

    def get_dataset_importer_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.importers.UnlabeledImageDatasetImporter`
        class for importing datasets of this type from disk.

        Returns:
            a :class:`fiftyone.utils.data.importers.UnlabeledImageDatasetImporter`
            class
        """
        return BiigleDatasetImporter

    def get_dataset_exporter_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.exporters.UnlabeledImageDatasetExporter`
        class for exporting datasets of this type to disk.

        Returns:
            a :class:`fiftyone.utils.data.exporters.UnlabeledImageDatasetExporter`
            class
        """
        return BiigleDatasetExporter
