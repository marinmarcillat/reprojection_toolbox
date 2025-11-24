from reprojection.geometry import fit_plane, project_points_to_plane
import pyvista as pv
import numpy as np
import os

def init_worker(points_data_arg, temp_dir_arg):
    # declare global variable
    global points_data, temp_dir
    # declare the global variable
    points_data = points_data_arg
    temp_dir = temp_dir_arg


def filter_points_task(args):
    global points_data, temp_dir

    ann_id = args["id"]
    camera_center = args["camera_center"]
    polygon_3d  = np.load(args["polygon_3D_file"], allow_pickle=True)

    print(f"Start filtering annotation {ann_id}")

    reprojection_plane_normal, reprojection_plane_center = fit_plane(polygon_3d)
    flatten_polygon = project_points_to_plane(polygon_3d, reprojection_plane_center, reprojection_plane_normal)
    #base_vert = [len(flatten_polygon)] + list(range(len(flatten_polygon)))
    #pv_flatten_polygon = pv.PolyData(flatten_polygon, base_vert)
    pv_flatten_polygon = pv.PolyData(flatten_polygon).delaunay_2d()
    reprojection_plane = pv.Plane(center=reprojection_plane_center, direction=reprojection_plane_normal, i_size=10,
                                  j_size=10)

    camera_direction = camera_center - np.array(reprojection_plane_center)
    camera_plane = pv.Plane(i_size=10, j_size=10, center = camera_center, direction = camera_direction)


    camera_direction = camera_center - np.array(reprojection_plane_center)

    # correct the reprojection plane normal
    if np.dot(reprojection_plane_normal, (camera_center - reprojection_plane_center)) > 0:
        reprojection_plane_normal = -reprojection_plane_normal

    #.triangulate()
    extruded_cam_cylinder = pv_flatten_polygon.extrude_trim(camera_direction, camera_plane)
    extruded_safety_cylinder = pv_flatten_polygon.extrude_trim(-reprojection_plane_normal, reprojection_plane.translate(reprojection_plane_normal*0.5))

    if extruded_cam_cylinder.extract_feature_edges(feature_edges=False, manifold_edges=False).n_cells != 0:
        extruded_cam_cylinder = extruded_cam_cylinder.fill_holes(1000).clean(tolerance=0.01).clean(tolerance=0.1)

    if extruded_safety_cylinder.extract_feature_edges(feature_edges=False, manifold_edges=False).n_cells != 0:
        extruded_safety_cylinder = extruded_safety_cylinder.fill_holes(1000).clean(tolerance=0.01).clean(tolerance=0.1)


    #, check_surface=False
    try:
        selected_1 = points_data.select_enclosed_points(extruded_cam_cylinder, tolerance=0.01)
        pts_1 = points_data.extract_points(
            selected_1['SelectedPoints'].view(bool),
            adjacent_cells=False,
        )
        selected_2 = points_data.select_enclosed_points(extruded_safety_cylinder, tolerance=0.01)
        pts_2 = points_data.extract_points(
            selected_2['SelectedPoints'].view(bool),
            adjacent_cells=False,
        )
    except RuntimeError:
        print("TP filtering failed")
        return ann_id, None

    filtered_points = np.array(pv.merge([pts_1, pts_2]).points)
    filepath = os.path.join(temp_dir, f"{ann_id}_tp3d.npy")
    np.save(filepath, filtered_points)
    return ann_id, filepath