from ifdo import iFDO
from ifdo.models import ImageSetHeader


import fiftyone.utils.data as foud
import fiftyone.types as fot
import eta.core.serial as etas
import fiftyone.core.labels as fol

import os
import json
import exrex

from ifdo.models import ImageAnnotation, AnnotationLabel, ImageSetHeader, ImageData

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


def open_ifdo(ifdo_path):
    return iFDO.load(ifdo_path)

def generate_empty_ifdo():
    uuid = exrex.getone('^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[4][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$|^[0-9a-fA-F]{12}4[0-9a-fA-F]{3}[89abAB][0-9a-fA-F]{15}$')
    ish = ImageSetHeader(
        image_set_name="",
        image_set_uuid=uuid,
        image_set_handle="",
    )
    isi = {}
    return iFDO(
        image_set_header=ish,
        image_set_items=isi
    )

class IFDODatasetExporter(foud.LabeledImageDatasetExporter, foud.ExportPathsMixin):

    def __init__(
        self,
        export_dir=None,
        data_path=None,
        labels_path=None,
        export_media=None,
        rel_dir=None,
        abs_paths=False,
        classes=None,
        include_confidence=None,
        include_attributes=None,
        image_format=None,
        pretty_print=True,
        ifdo = None,
        name="",
        handle="",
    ):
        data_path, export_media = self._parse_data_path(
            export_dir=export_dir,
            data_path=data_path,
            export_media=export_media,
            default="data/",
        )

        labels_path = self._parse_labels_path(
            export_dir=export_dir,
            labels_path=labels_path,
            default="labels.json",
        )

        super().__init__(export_dir=export_dir)

        self.data_path = data_path
        self.labels_path = labels_path
        self.export_media = export_media
        self.rel_dir = rel_dir
        self.abs_paths = abs_paths
        self.classes = classes
        self.include_confidence = include_confidence
        self.include_attributes = include_attributes
        self.image_format = image_format
        self.pretty_print = pretty_print

        self._labels_map_rev = None
        self._media_exporter = None
        self._labels_dict = None

        if ifdo is None:
            self.ifdo = generate_empty_ifdo()
            self.ifdo.image_set_header.image_set_name = name
            self.ifdo.image_set_header.image_set_handle = handle
        else:
            self.ifdo = ifdo


    @property
    def requires_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return fol.Detections, fol.Polylines

    def setup(self):
        self._labels_dict = {}
        self._parse_classes()

        self._media_exporter = foud.ImageExporter(
            self.export_media,
            export_path=self.data_path,
            rel_dir=self.rel_dir,
            default_ext=self.image_format,
            ignore_exts=True,
        )
        self._media_exporter.setup()

    def export_sample(self, image_or_path, label, metadata):
        out_image_path, uuid = self._media_exporter.export(image_or_path)

        w, h = metadata.width, metadata.height

        key = out_image_path if self.abs_paths else uuid
        image = ImageData()

        converted_detections = []
        if isinstance(label, fol.Polylines):
            polylines = label.polylines
            for polyline in polylines:
                coordinates = pl_relative_to_absolute(polyline.points, w, h, polyline.closed)
                label = AnnotationLabel(label=polyline.label, annotator="fiftyone")
                annotation = ImageAnnotation(coordinates=coordinates, labels=[label], shape='polygon')
                converted_detections.append(annotation)

            image.image_annotations = converted_detections

        elif isinstance(label, fol.Detections):
            detections = label.detections

            for detection in detections:
                coordinates = bb_relative_to_absolute(detection.bounding_box, w, h)
                label = AnnotationLabel(label=detection.label, annotator="fiftyone", confidence=detection.confidence)
                annotation = ImageAnnotation(coordinates=coordinates, labels=[label], shape='rectangle')
                converted_detections.append(annotation)
            image.image_annotations = converted_detections
        self.ifdo.image_set_items[key] = [image]

    def close(self, *args):
        self.ifdo.image_set_header.image_annotation_labels = self.classes
        result = self.ifdo.to_dict()
        etas.write_json(
            result, self.labels_path, pretty_print=self.pretty_print
        )

        self._media_exporter.close()

    def _parse_classes(self):
        if self.classes is not None:
            self._labels_map_rev = foud._to_labels_map_rev(self.classes)


# TODO: Implement the importer
class IFDODatasetImporter(foud.LabeledImageDatasetImporter, foud.ImportPathsMixin):
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
        dataset_dir,
        ifdo_path = None,
        shuffle=False,
        seed=None,
        max_samples=None,
        **kwargs,
    ):
        
        self.data_path = self._parse_data_path(
            dataset_dir=dataset_dir,
            data_path=ifdo_path,
            default="data/",
        )

        if ifdo_path is None:
            ifdo_path = os.path.join(dataset_dir, "labels.json")
        
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        with open(ifdo_path) as file:
            json_data = json.load(file)

        self.ifdo = iFDO.from_dict(json_data)

        self._info = None
        self._classes = None
        self._labels_paths_map = None
        self._labels_map = None
        self._filepaths = None
        self._iter_filepaths = None
        self._num_samples = None

    def __iter__(self):
        self._iter_filepaths = iter(self._filepaths)
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        filepath = next(self._iter_filepaths)

        labels_path = self._labels_paths_map.get(filepath, None)
        if labels_path:
            # Labeled image
            label = []
            #load_yolo_annotations(
            #    labels_path, self._classes, label_type=self.label_type
            #)
        else:
            # Unlabeled image
            label = None

        return filepath, None, label

    @property
    def has_dataset_info(self):
        return True

    @property
    def has_image_metadata(self):
        return False

    @property
    def label_cls(self):
        return (fol.Detections, fol.Polylines)

    def setup(self):
        ifdo = open_ifdo(self.ifdo_path)

        # set the setup here

    def get_dataset_info(self):
        return self._info

    def close(self, *args):
        pass


class IFDOImageDataset(fot.LabeledImageDataset):
    """Custom unlabeled image dataset type."""

    def get_dataset_importer_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.importers.UnlabeledImageDatasetImporter`
        class for importing datasets of this type from disk.

        Returns:
            a :class:`fiftyone.utils.data.importers.UnlabeledImageDatasetImporter`
            class
        """
        return IFDODatasetImporter

    def get_dataset_exporter_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.exporters.UnlabeledImageDatasetExporter`
        class for exporting datasets of this type to disk.

        Returns:
            a :class:`fiftyone.utils.data.exporters.UnlabeledImageDatasetExporter`
            class
        """
        return IFDODatasetExporter
