import Metashape
import numpy as np
import os
from ast import literal_eval
from tqdm import tqdm
import pandas as pd
import reprojection.metashape_utils as mu
import reprojection.geometry as geometry
import pyvista as pv



class CameraReprojector:

    def __init__(self, camera: Metashape.Camera, chunk: Metashape.Chunk, model: Metashape.Model, camera_files_dir):
        self.chunk = chunk
        self.camera = camera
        self.model = model

        self.distance_filter = 20
        self.decimation_factor = 20

        self.h = int(self.camera.photo.meta['File/ImageHeight'])
        self.w = int(self.camera.photo.meta['File/ImageWidth'])

        self.annotations = []

        self.contour_file = os.path.join(camera_files_dir, f"{camera.label}.ply")

        if not os.path.exists(self.contour_file):
            self.contour = self.get_contour()
            if self.contour is None:
                raise Exception("No contour found")
            self.contour.save(self.contour_file)
        else:
            self.contour = pv.read(self.contour_file)


    def check_inv_reproj_inbound(self, point):
        x, y = point[0], point[1]
        return 0 <= x <= self.w and 0 <= y <= self.h

    def inverse_reprojection(self, point):
        coords_2D = self.camera.project(point)
        if coords_2D is not None and self.check_inv_reproj_inbound(coords_2D):
            return np.array(coords_2D)
        return None

    def inverse_reproject_polygon(self, polygon_3d):
        res = []
        for point in polygon_3d:
            p = self.inverse_reprojection(point)
            if p is not None:
                res.append(p)
        return res or None


    def reproject_point(self, point):
        coords_2d = Metashape.Vector(point)
        unprojected_point = self.camera.unproject(coords_2d)
        t = self.model.pickPoint(self.camera.center, unprojected_point)
        if t is not None:
            repr_point = np.array(t)
            dist = np.linalg.norm(np.array(unprojected_point)-repr_point)
            if dist <= self.distance_filter:
                return repr_point
        return None

    def reproject_polygon(self, polygon):
        polygon = [self.reproject_point(point) for point in polygon]
        polygon = np.array([point for point in polygon if point is not None])
        if len(polygon) > 2:
            filtered = geometry.eccentricity_filter(polygon, np.array(self.camera.center))
            filtered = geometry.sor_filter(filtered)
            if len(filtered) > 2:
                return filtered

    def local2world(self, point_local):
        point_local = Metashape.Vector(point_local)
        point3d_world = self.chunk.crs.project(self.chunk.transform.matrix.mulp(point_local))
        return np.array(point3d_world)

    def world2local(self, point_world):
        point_world = Metashape.Vector(point_world)
        point_internal = self.chunk.transform.matrix.inv().mulp(self.chunk.crs.unproject(point_world))
        return np.array(point_internal)

    def get_contour(self):
        point_list =  mu.get_cameras_tie_points(self.chunk, [self.camera])
        if point_list is not None:
            ph = geometry.get_polyhull(point_list)
            return ph

    def check_contact(self, other_cam):
        if self.contour is not None and other_cam.contour is not None:
            return geometry.check_contact(self.contour, other_cam.contour)
        return False

    def get_cameras_in_view_annotation(self, points, cameras: list):
        cameras_in_view = []
        polygon = Biigle_polygon_to_polygon(points)
        polygon_3d = self.reproject_polygon(polygon)

        if polygon_3d is not None:
            cameras_in_view_camera = self.get_cameras_in_view_camera(cameras)
            cameras_in_view.extend(
                other_camera
                for other_camera in cameras_in_view_camera
                if other_camera.inverse_reproject_polygon(polygon_3d) is not None
            )
        return cameras_in_view

    def get_cameras_in_view_camera(self, cameras: list):
        cameras_in_view = []
        for other_camera in cameras:
            dist = np.linalg.norm(np.array(self.camera.center)-np.array(other_camera.camera.center))
            if dist < 50 and dist != 0:# First distance filter
                if self.check_contact(other_camera): # Second contact filter
                    cameras_in_view.append(other_camera)
        return cameras_in_view

    def plot_polygon(self, polygon, label):
        # polygon: list of points
        if not self.chunk.shapes:
            self.chunk.shapes = Metashape.Shapes()
            self.chunk.shapes.crs = self.chunk.crs

        poly_local = self.reproject_polygon(polygon)
        if poly_local is not None:
            poly_world = [self.local2world(i) for i in poly_local]
            if len(poly_world) >=3:
                shape = self.chunk.shapes.addShape()
                shape.label = label
                shape.geometry = Metashape.Geometry.Polygon(poly_world)

    def plot_polygon3D(self, polygon, label):
        if polygon is not None:
            if not self.chunk.shapes:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            if len(polygon) >=3:
                poly_world = [np.array(self.chunk.crs.project(self.chunk.transform.matrix.mulp(Metashape.Vector(i)))) for i in
                              polygon]
                shape = self.chunk.shapes.addShape()
                shape.label = str(label)
                shape.geometry = Metashape.Geometry.Polygon(poly_world)

    def plot_point(self, point, label, group = None):
        if point is not None:
            if not self.chunk.shapes:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            if len(point) ==3:
                point_world = np.array(self.chunk.crs.project(self.chunk.transform.matrix.mulp(Metashape.Vector(point))))
                shape = self.chunk.shapes.addShape()
                shape.label = str(label)
                shape.geometry = Metashape.Geometry.Point(point_world)
                if group is not None and type(group) is Metashape.ShapeGroup:
                    shape.group = group


def Biigle_polygon_to_polygon(polygon):
    if len(polygon) == 3:
        return geometry.circle_to_polygons(polygon[:2], polygon[2])
    coords = list(zip(*[iter(polygon)] * 2))
    return geometry.rectangle_to_polygons(coords) if len(polygon) == 8 else coords

def meta_polygon_to_biigle(polygon):
    coords = []
    for point in polygon:
        coords.extend(point)
    return coords

def get_polygon_and_3d_poly(points, camera: CameraReprojector):
    polygon = np.array(Biigle_polygon_to_polygon(points))
    polygon_3d = np.array(camera.reproject_polygon(polygon))
    if polygon.shape != () and polygon_3d.shape != ():
        return polygon, polygon_3d
    return None, None

def get_camera(filename, cameras: list[CameraReprojector]):
    for camera in cameras:
        if filename == camera.camera.label:
            return camera


if __name__ == "__main__":

    report = pd.read_csv(r"D:\Chereef_metashape\2022\Coral Garden\annotations\inference.csv")
    report['points'] = report.points.apply(lambda x: literal_eval(str(x)))

    doc = Metashape.Document()
    doc.open(r"D:\Chereef_metashape\2022\Coral Garden\coral_garden_22.psx")
    chunk = doc.chunk

    cameras = [CameraReprojector(chunk, camera) for camera in chunk.cameras if camera.transform]

    print(f"Len: {len(cameras)}")

    for photo in tqdm(report.filename.unique()):
        camera = get_camera(photo[:-4], cameras)
        annotations = report[report.filename == photo]
        for id, annotation in annotations.iterrows():
            camera.plot_polygon(Biigle_polygon_to_polygon(annotation.points), annotation.label_name)

    doc.save()









    #doc.save()


