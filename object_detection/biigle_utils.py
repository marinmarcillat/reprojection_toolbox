from dataclasses import dataclass
import pickle
import os
from datetime import datetime, timedelta
from tqdm import tqdm


class Video:
    def __init__(self, video_id: int, volume, api):
        self.video_id = video_id
        self.api = api
        self.volume = volume

        response = self.api.get(f'videos/{video_id}').json()
        self.filename = response['filename']
        self.duration = response['duration']

        self.start = get_video_start(self)

        self.abs_path = os.path.join(self.volume.dir, self.filename)
        self.annotations = self.get_annotations()

    def get_annotations(self):
        annotations = self.api.get(f'videos/{self.video_id}/annotations').json()
        return [VideoAnnotation(
            ann["id"], self, ann['frames'],
            ann['points'], ann["labels"][0]["label"]["name"],
            ann["labels"][0]["label_id"], ann['shape_id']
        )
                for ann in annotations if ann['shape_id'] != 7]

    def add_new_video_annotation(self, video_annotation):
        req = {
                "shape_id": video_annotation.shape_id,
                "label_id": video_annotation.label_id,
                "frames": video_annotation.frames,
                "points": video_annotation.points
            }
        response = self.api.post(f'videos/{self.video_id}/annotations', json=req).json()

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(name={self.filename!r}, id={self.video_id!r})')

class VideoVolume:
    def __init__(self, volume_id, api):
        self.volume_id = volume_id
        self.api = api

        self.videos_ids = self.api.get(f'volumes/{self.volume_id}/files').json()

        response = self.api.get(f'volumes/{self.volume_id}').json()
        self.dir = response['url'].split("//")[1]

        self.videos = [Video(i, self, self.api) for i in tqdm(self.videos_ids)]

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(id={self.volume_id!r}, videos={self.videos!r})')

@dataclass
class VideoAnnotation:
    ann_id: int
    video: Video
    frames: list[float]
    points: list[list[float]]
    label: str
    label_id: int
    shape_id: int

    def __post_init__(self):
        self.tracked = (len(self.frames) != 1)
        self.keyframe = self.frames[0]
        self.keyframe_point = self.points[0]
        self.frames_abs = [self.video.start + timedelta(seconds = frame) for frame in self.frames]
        self.keyframe_abs = self.frames_abs[0]


def get_video_start(video: Video, pattern = "%y%m%d%H%M%S" ):
    return datetime.strptime(video.filename.split('_')[2], pattern)


def get_unique_keyframes_abs(video_ann: list[VideoAnnotation]):
    kf = list(set([ann.keyframe_abs for ann in video_ann]))
    kf.sort()
    return kf

def get_unique_keyframes(video_ann: list[VideoAnnotation]):
    kf = list(set([ann.keyframe for ann in video_ann]))
    kf.sort()
    return kf

def filter_video_annotation(annotations, keys: list[dict]):
    for key in keys:
        if key['name'] == 'shape':
            annotations = [ann for ann in annotations if ann.shape_id in key['values']]
        if key['name'] == 'label_name':
            annotations = [ann for ann in annotations if ann.label in key['values']]
        if key['name'] == 'keyframe':
            annotations = [ann for ann in annotations if ann.keyframe in key['values']]
    return annotations


class Image:
    def __init__(self, image_id: int, volume, api, vol_filenames=None):
        self.image_id = image_id
        self.volume = volume
        self.api = api

        self.annotations = None

        if vol_filenames is None:
            response = self.api.get(f'images/{self.image_id}').json()
            self.filename = response['filename']
        else:
            self.filename = vol_filenames[str(self.image_id)]

        self.abs_path = os.path.join(self.volume.dir, self.filename)


    def get_annotations(self):
        annotations = self.api.get(f'images/{self.image_id}/annotations').json()
        self.annotations = [ImageAnnotation(
            ann['id'], self,
            ann['points'], ann["labels"][0]["label"]["name"],
            ann["labels"][0]["label_id"], ann['shape_id']
        )
            for ann in annotations]
        return self.annotations

    def add_annotation(self, annotation_list: list):
        req = []
        for ann in annotation_list:
            req.append({
                "image_id": self.image_id,
                "shape_id": ann.shape_id,
                "label_id": ann.label_id,
                "points": ann.points,
                "confidence": 1
            })
        chunks = [req[x:x + 100] for x in range(0, len(req), 100)]
        for chunk in chunks:
            response = self.api.post(f'image-annotations', json=chunk)


class ImageVolume:
    def __init__(self, volume_id, api):
        self.volume_id = volume_id
        self.api = api
        response = self.api.get(f'volumes/{self.volume_id}').json()
        self.dir = response['url'].split("//")[1]

        self.image_ids = self.get_images_ids()
        self.image = self.get_images()

    def get_images_ids(self):
        return self.api.get(f'volumes/{self.volume_id}/files').json()

    def get_images(self):
        volume_filenames = self.api.get(f'volumes/{self.volume_id}/filenames').json()
        return [Image(i, self, self.api, volume_filenames) for i in self.image_ids]

    def add_image(self, img: str):
        req = {"images": img}
        response = self.api.post(f'volumes/{self.volume_id}/files', json=req).json()

        self.image_ids = self.get_images_ids()
        self.image = self.get_images()
        return response

    def request_sam(self):
        for id in tqdm(self.image_ids):
            res = self.api.post(f'images/{id}/sam-embedding').json()


@dataclass
class ImageAnnotation:
    ann_id: int
    image: Image
    points: list[float]
    label: str
    label_id: int
    shape_id: int


def get_frame_image(keyframe: float, video: Video, image_list: list[Image]):
    image_filename = f"{video.video_id}_{keyframe}.png"
    for img in image_list:
        if img.filename == image_filename:
            return img
    return None

def vid2image_annotation(vid_ann_list: list[VideoAnnotation], img: Image):
    res = []
    for ann in vid_ann_list:
        res.append(ImageAnnotation(-1, img, ann.keyframe_point, ann.label, ann.label_id, ann.shape_id))
    return res

def save_annotations(annotation_list, file):
    with open(file, "wb") as fp:  # Pickling
        pickle.dump(annotation_list, fp)



