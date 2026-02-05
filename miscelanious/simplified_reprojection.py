import Metashape
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

project_path = r"F:\01-Reconstructions\mtshp_projects\combined\main.psx"
annotation_report_path = r"D:\03-Change_detection\colonies\annotations_reports\676-coral-cg-chereef25.csv"

doc = Metashape.Document()
doc.open(project_path)
chunk = doc.chunk
model = doc.chunk.model

meta_cameras = [camera for camera in chunk.cameras if camera.transform]

inference_report = pd.read_csv(annotation_report_path)
inference_report = inference_report[["filename", "label_name", "points"]]
if type(inference_report.points[0]) == str:
    inference_report['points'] = inference_report.points.apply(lambda x: literal_eval(str(x)))

def reproject_point(model, camera, point):
    coords_2d = Metashape.Vector(point)
    unprojected_point = camera.unproject(coords_2d)
    t = model.pickPoint(camera.center, unprojected_point)
    if t is not None:
        repr_point = np.array(t)
        return repr_point
    return None

def get_camera(filename, cameras):
    for camera in cameras:
        if filename == camera.label:
            return camera

result = []
photo_id = 0
for photo in tqdm(inference_report.filename.unique()):
    camera = get_camera(photo, meta_cameras)
    if camera is None:
        print(f"Skipping {photo} because not in model")
        continue
    annotations = inference_report[inference_report.filename == photo]
    for id, annotation in annotations.iterrows():
        points = np.array(annotation.points)
        if len(points) == 3:
            center = points[:2]
        else:
            coords = list(zip(*[iter(points)] * 2))
            center = np.mean(coords, axis=0)
        repr_point = reproject_point(model, camera, center)
        if repr_point is not None:
            print(f"Reprojected point for {photo}: {repr_point}")
            result.append([repr_point, f"{annotation.label_name}_{photo_id}"])
        else:
            print(f"Failed to reproject point for {photo}")
    photo_id += 1

if not chunk.shapes:
    chunk.shapes = Metashape.Shapes()
    chunk.shapes.crs = chunk.crs

pd_result = pd.DataFrame(result, columns=["point", "label"])
for id, annotation in pd_result.iterrows():
    point = annotation.point
    label = annotation.label
    if len(point) == 3:
        point_world = np.array(chunk.crs.project(chunk.transform.matrix.mulp(Metashape.Vector(point))))
        shape = chunk.shapes.addShape()
        shape.label = str(label)
        shape.geometry = Metashape.Geometry.Point(point_world)

doc.save()

