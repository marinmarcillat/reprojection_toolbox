import reprojection.reprojection as reprojection
import Metashape
import pandas as pd
import reprojection.geometry as geometry
import numpy as np
from tqdm import tqdm
import pyvista as pv
import reprojection.reprojection_database as rdb
from ast import literal_eval
import reprojection.camera_utils as cu
import os, math

def save_3d_polygon(poly_3d, save_dir, annotation_id):
    filepath = os.path.join(save_dir, f"{annotation_id}_annotation_p3d.npy")
    np.save(filepath, poly_3d)
    return filepath

def save_tie_points(tie_points, save_dir, annotation_id):
    filepath = os.path.join(save_dir, f"{annotation_id}_tp3d.npy")
    np.save(filepath, tie_points)
    return filepath

def save_polygon(poly, save_dir, annotation_id, camera_name):
    filepath = os.path.join(save_dir, f"{annotation_id}_{camera_name}_ir.npy")
    np.save(filepath, poly)
    return filepath

def get_near_cameras(session, camera: rdb.Camera, threshold = 10):
    x, y, z = camera.center_x, camera.center_y, camera.center_z
    filtered = []
    filtered.extend(
        c
        for c in session.query(rdb.Camera).all()
        if np.linalg.norm(
            np.array([c.center_x, c.center_y, c.center_z])
            - np.array([x, y, z])
        )
        < threshold
    )
    return filtered

def filter_cam_on_contact(session, camera_name_list: list[str], camera: rdb.Camera):
    center_ph = pv.read(camera.poly_hull_file)
    contacts = []
    for c in session.query(rdb.Camera).filter(rdb.Camera.name.in_(camera_name_list)).all():
        other_ph = pv.read(c.poly_hull_file)
        if geometry.check_contact(center_ph, other_ph):
            contacts.append(c.name)
    return contacts

def check_polygon_contact(ir, other_ir):
    if ir is None or other_ir is None:
        return False
    if os.path.exists(ir.inv_reproj_file) and os.path.exists(other_ir.inv_reproj_file):
        sh1 = np.load(ir.inv_reproj_file, allow_pickle=True)
        sh2 = np.load(other_ir.inv_reproj_file, allow_pickle=True)
        if sh1.shape != () and sh2.shape != () and len(sh1) > 2 and len(sh2) > 2:
            return geometry.check_overlap(sh1, sh2)
    return False

def pre_check_bounding_box(a1, a2):
    poly_3d_1 = np.load(a1.polygon_3D_file, allow_pickle=True)
    poly_3d_2 = np.load(a2.polygon_3D_file, allow_pickle=True)

    min1 = poly_3d_1.min(axis=0)
    min2 = poly_3d_2.min(axis=0)
    max1 = poly_3d_1.max(axis=0)
    max2 = poly_3d_2.max(axis=0)

    return no_overlap(np.c_[min1, max1], np.c_[min2, max2])

def no_overlap(box1, box2, count_edge=False):
    """
    Vectorized bounding box overlap check.

    Args:
        box1 (np.ndarray): First bounding box with min and max coordinates
        box2 (np.ndarray): Second bounding box with min and max coordinates
        count_edge (bool, optional): Whether to count edge touching as overlap. Defaults to False.

    Returns:
        bool: True if boxes do not overlap, False otherwise
    """
    if count_edge:
        return np.any(box1[:, 0] > box2[:, 1]) or np.any(box2[:, 0] > box1[:, 1])
    else:
        return np.any(box1[:, 0] >= box2[:, 1]) or np.any(box2[:, 0] >= box1[:, 1])


def no_overlap_1d(min1,max1,min2,max2,count_edge=False):
    if count_edge:
        return min1>max2 or min2>max1
    else:
        return min1>=max2 or min2>=max1

def check_overlap(annotation1, annotation2):
    if annotation1 == annotation2:
        return False

    if pre_check_bounding_box(annotation1, annotation2): # Check bounding box overlap first
        return False

    tp1 = np.load(annotation1.tie_point_file)
    tp2 = np.load(annotation2.tie_point_file)

    i = geometry.get_nb_common_points(tp1, tp2)
    u = min([len(tp1),len(tp2)])
    if u == 0 or i == 0:
        return False
    iou = i/u

    return iou > 0.1

def inference_report_to_reprojection_database(db_dir, inference_report, cameras_reprojectors, tie_points, db_name="reprojection"):
    if type(inference_report.points[0]) == str:
        inference_report['points'] = inference_report.points.apply(lambda x: literal_eval(str(x)))

    session, _ = rdb.open_reprojection_database_session(db_dir, True, db_name)

    temp_dir = os.path.join(db_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print("adding")
    for photo in tqdm(inference_report.filename.unique()):
        reproj_camera = reprojection.get_camera(photo[:-4], cameras_reprojectors)
        if reproj_camera is None:
            print(f"Skipping {photo} because not in model")
            continue
        if reproj_camera.contour is None:
            print(f"Skipping {photo} because missing contour")
            continue
        db_camera = cu.camera_reprojector_to_db(session, reproj_camera)

        annotations = inference_report[inference_report.filename == photo]
        for id, annotation in annotations.iterrows():
            a = rdb.Annotation(label = annotation.label_name, camera_name=db_camera.name)
            if "confidence" in list(annotations.columns):
                a.confidence = annotation.confidence
            session.add(a)
            session.commit()
            poly3d_file = os.path.join(temp_dir, f"{a.id}_annotation_p3d.npy")
            tie_point_file = os.path.join(temp_dir, f"{a.id}_tp3d.npy")
            if not (os.path.exists(poly3d_file) and os.path.exists(tie_point_file)):
                polygon, polygon_3d = reprojection.get_polygon_and_3d_poly(annotation.points, reproj_camera)
                if polygon_3d is not None and polygon is not None:  # Check if polygon is not none
                    filtered_tie_points = geometry.filter_tie_points(polygon_3d, reproj_camera, tie_points)
                    poly3d_file = save_3d_polygon(polygon_3d, temp_dir, a.id)
                    tie_point_file = save_tie_points(filtered_tie_points, temp_dir, a.id)
                else:
                    session.delete(a) # Remove annotation if no valid polygon
                    continue
            a.polygon_3D_file = poly3d_file
            a.tie_point_file = tie_point_file
    session.commit()
    return session

def annotations_to_individuals(session, db_dir):

    temp_dir = os.path.join(db_dir, "temp")
    if not os.path.exists(temp_dir):
        raise FileNotFoundError(f"Temp directory {temp_dir} does not exist")

    print("Get individuals")
    for camera in tqdm(session.query(rdb.Camera).all()):
        annotations = session.query(rdb.Annotation).filter_by(camera_name=camera.name).all()
        if len(annotations) != 0:
            near_cams = [c.name for c in get_near_cameras(session, camera)]
            for annotation in annotations:
                if annotation.individual_id is None: # Check if annotation has been attributed to individual
                    near_annotations = session.query(rdb.Annotation).filter_by(label = annotation.label).filter(rdb.Annotation.camera_name.in_(near_cams)).all()
                    filtered_cams = filter_cam_on_contact(session, near_cams, camera)
                    filtered_annotations = [annotation for annotation in near_annotations if annotation.camera_name in filtered_cams]
                    same_ind = [
                        other_annotation
                        for other_annotation in filtered_annotations
                        if check_overlap(annotation, other_annotation)
                    ]
                    # Check if other annotations have been attributed to individual
                    ind_list = list(set([a.individual_id for a in same_ind if a.individual_id is not None]))
                    if len(ind_list) != 0: # If there are other annotations attributed to individuals
                        if len(ind_list) == 1: # If there is only 1 individual
                            ind = session.query(rdb.Individual).filter_by(id=ind_list[0]).first() # Get individual
                        elif len(ind_list) > 1: # If there are more than 1 individual, create new individual
                            print(f"More than 1 individual: {ind_list}")
                            ind = rdb.Individual()
                            session.add(ind)
                            session.commit()
                            print(f"New individual: {ind.id}")
                        # Update all annotations to same individual
                        extra_anns = session.query(rdb.Annotation).filter_by(label = annotation.label).filter(rdb.Annotation.individual_id.in_(ind_list)).all()
                        for a in extra_anns:
                            a.individual_id = ind.id
                    else: # If no other annotations have been attributed to individuals, create new individual
                        ind = rdb.Individual()
                        session.add(ind)
                        session.commit()
                    # Update all annotations to individual
                    annotation.individual_id = ind.id
                    for a in same_ind:
                        a.individual_id = ind.id
                    session.commit()
    return session

if __name__ == "__main__":
    db_dir = r"D:\chereef_marin\chereef22"
    create_db = False

    report = pd.read_csv(r"D:\chereef_marin\chereef22\annotations\inference_multilabel_SAHI.csv")
    report['points'] = report.points.apply(lambda x: literal_eval(str(x)))

    doc = Metashape.Document()
    doc.open(r"D:\chereef_marin\chereef_2223\coral_garden_2223.psx")
    chunk = doc.chunks[0] #2022
    model = chunk.models[1]

    temp_dir = r"D:\chereef_marin\chereef22\temp"


    cameras_reprojectors = cu.chunk_to_camera_reprojector(chunk, db_dir)


    if create_db:
        session = inference_report_to_reprojection_database(db_dir, report, cameras_reprojectors, "reprojection")

    else:
        print("Getting DB from existing")
        session, path = rdb.open_reprojection_database_session(db_dir, False)

    session = annotations_to_individuals(session)


    doc.save()

    print("stop")
    print("done")



