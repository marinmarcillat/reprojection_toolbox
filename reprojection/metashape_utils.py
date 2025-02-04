import Metashape
from shutil import copy2
import reprojection.reprojection as reprojection
import numpy as np
from tqdm import tqdm
import reprojection.reprojection_database as rdb
import reprojection.camera_utils as cu
import os


def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]


def add_images(chunk, image_dir):
    photos = find_files(image_dir, [".jpg", ".jpeg", ".tif", ".tiff"])
    chunk.addPhotos(photos)

    print('Analyzing images, Removing images with quality score < 0.5')
    chunk.analyzeImages()
    for c in chunk.cameras:
        if float(c.meta['Image/Quality']) < 0.5:
            print(f"Removed image {c.label} with score {c.meta['Image/Quality']}")
            c.enabled = False

def get_sparse_model(chunk):
    models = chunk.models
    for model in models:
        if int(model.meta['BuildModel/source_data']) == 0:
            return model
    print("No sparse model found, creating one...")
    chunk.buildModel(source_data=Metashape.DataSource.TiePointsData)
    return chunk.models[-1]

def get_overlapping_images(chunk, export_dir):

    chunk = chunk.copy()

    chunk.model = get_sparse_model(chunk)
    chunk.reduceOverlap(overlap = 1)

    meta_cameras_path = [camera.photo.path for camera in chunk.cameras if camera.transform and camera.enabled]

    for camera_path in meta_cameras_path:
        copy2(camera_path, export_dir)

    return chunk  # To be destroyed after use with doc.remove([chunk])


def get_cameras_tie_points(chunk, cameras: list):
    point_cloud = chunk.tie_points
    points = point_cloud.points
    npoints = len(points)

    point_list = []
    for photo in cameras:
        point_index = 0
        for proj in point_cloud.projections[photo]:
            track_id = proj.track_id
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
                if points[point_index].valid:
                    point_list.append(list(points[point_index].coord)[:3])
    if len(point_list) > 0:
        return point_list

def plot_reprojection_db(session, chunk, group_label, db_dir, confidence = 0.5):
    cameras_reprojectors = cu.chunk_to_camera_reprojector(chunk, db_dir)

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs
    group = chunk.shapes.addGroup()
    group.label = group_label


    for individuals in tqdm(session.query(rdb.Individual).all()):
        annotations = session.query(rdb.Annotation).filter_by(individual_id=individuals.id).filter(rdb.Annotation.confidence>confidence).all()
        poly_3d_list = [np.load(annotation.polygon_3D_file, allow_pickle=True) for annotation in annotations if annotation is not None]
        if len(poly_3d_list) == 0:
            continue
        poly_3d_coords = np.concatenate(poly_3d_list, axis=0)
        centroid = np.mean(poly_3d_coords, axis=0)
        if len(centroid) == 3:
            camera = session.query(rdb.Camera).filter_by(name=annotations[0].camera_name).first()
            camera_reproj = reprojection.get_camera(camera.name, cameras_reprojectors)
            camera_reproj.plot_point(centroid, f"{annotations[0].label}", group)


def plot_all_annotations(session, cameras_reprojectors):
    print("plot all individuals")
    for individual in tqdm(session.query(rdb.Individual).all()):
        annotations = session.query(rdb.Annotation).filter_by(individual_id=individual.id).all()
        for annotation in annotations:
            if annotation is not None:
                poly_3d = np.load(annotation.polygon_3D_file, allow_pickle=True)
                if poly_3d.shape != ():
                    camera = session.query(rdb.Camera).filter_by(name=annotation.camera_name).first()
                    camera_reproj = reprojection.get_camera(camera.name, cameras_reprojectors)
                    camera_reproj.plot_polygon3D(poly_3d, f"{annotation.label}")
                    continue

def export_shapes(chunk, export_dir, group_label, epsg = "EPSG::32629", shift = [0, 0, 0]):
    if not chunk.shapes:
        print("No shapes found")
        return
    group = None
    for g in chunk.shapes.groups:
        if g.label == group_label:
            group = g
            break
    if group is None:
        print("No group found")
        return
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    chunk.exportShapes(
        os.path.join(export_dir, f"{group_label}.shp"),
        group = [group.key],
        save_points = True,
        crs = Metashape.CoordinateSystem(epsg),
        format = Metashape.ShapesFormatSHP,
        shift = Metashape.Vector(shift))

# chunk = Metashape.app.document.chunk


