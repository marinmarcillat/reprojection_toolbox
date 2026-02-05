import pyvista as pv
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def cut_with_plane(pcd, plane_normal, plane_center):
    left = pcd.clip(normal = plane_normal, origin = plane_center, invert = True, progress_bar=True)
    right = pcd.clip(normal = plane_normal, origin = plane_center, invert = False, progress_bar=True)
    return left, right

def get_plane_equation(pc):
    plane= pv.fit_plane_to_points(pc, return_meta=True)
    return plane[1], plane[2]


def rotate_pcd(pc, center, normal):
    target_normal = np.array([0, 0, 1])
    # Normalize vectors
    plane_normal = normal / np.linalg.norm(normal)
    target_normal = target_normal / np.linalg.norm(target_normal)

    axis = np.cross(plane_normal, target_normal)
    angle = np.arccos(np.clip(np.dot(plane_normal, target_normal), -1.0, 1.0))

    rot = pc.copy()
    rot.rotate_vector(vector=axis, angle=np.rad2deg(angle), point=center, inplace=True)
    return rot

def rotate_matrix(pc, trsfm_matrix):
    t = pv.Transform(trsfm_matrix, point = pc.center).invert()
    rot = pc.copy()
    rot.transform(t, inplace=True)
    return rot

def pvtopandas(pcd):
    df = pd.DataFrame(pcd.points, columns=['x', 'y', 'z'])
    for key in pcd.array_names:
        df[key] = pcd[key]
    return df

def load_faces(plane_export_dir):
    faces = {}
    for name in ["left", "right"]:
        plane_data = np.load(os.path.join(plane_export_dir, f"{name}_plane.npy"), allow_pickle=True).item()
        faces[name] = {"center": plane_data["center"], "normal": plane_data["normal"]}
    return faces

def compute_faces(pcd_file, plane_export_dir, cut_plane_normal, cut_plane_center):
    pcd = pv.read(pcd_file, progress_bar=True)
    pc_left, pc_right = cut_with_plane(pcd, cut_plane_normal, cut_plane_center)

    faces = {}
    for pc, name in zip([pc_left, pc_right], ["left", "right"]):
        center, normal = get_plane_equation(pc)
        print(f"{name} plane center: {center}, normal: {normal}")
        save_path = os.path.join(plane_export_dir, f"{name}_plane.npy")
        np.save(save_path, {"center": np.array(center), "normal": np.array(normal)})
        faces[name] = {"center": center, "normal": normal}
    return faces

def split_rotate_annotations(annotations_pcd, cut_plane_normal, cut_plane_center, faces):
    annotations_left, annotations_right = cut_with_plane(annotations_pcd, cut_plane_normal, cut_plane_center)
    res = {}
    for ann, name in zip([annotations_left, annotations_right], ["left", "right"]):
        rot = rotate_pcd(ann, faces[name]["center"], faces[name]["normal"])
        res[name] = pvtopandas(rot)
    return res


if __name__ == "__main__":
    print("starting")
    pcd_file = r"D:\98_work\ful_vw_model\model.ply"
    plane_export_dir = r"D:\98_work\ful_vw_model"
    annotations_pcd = pv.read(pcd_file, progress_bar=True)

    cut_plane_normal = [-0.527798, 0.847633, -0.0542893]
    cut_plane_center = [1045.463257, 205.031754, -809.335571]

    pcd = pv.read(pcd_file, progress_bar=True)
    pc_left, pc_right = cut_with_plane(pcd, cut_plane_normal, cut_plane_center)

    for pc, name in zip([pc_left, pc_right], ["left", "right"]):
        center, normal = get_plane_equation(pc)
        rotated = rotate_pcd(pc, center, normal)
        filename = os.path.join(plane_export_dir, f"{name}_part.ply")
        rotated.save(filename)


