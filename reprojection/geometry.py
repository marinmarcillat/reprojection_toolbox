import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from scipy.spatial import ConvexHull
import pyvista as pv
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import geopandas as gpd
import pandas as pd
import random
import os


def offset(c, d, bearing):
    x = c[0] + math.sin(bearing) * d
    y = c[1] + math.cos(bearing) * d
    return [x, y]

def circle_to_polygons(center, radius, edges=32):
    coordinates = []
    for i in range(edges):
        coordinates.append(offset(center, radius, (2 * math.pi * i) / edges))
    return coordinates

def rectangle_to_polygons(coords, edges=20):
    polygon = [coords[0]]
    for i in [0, 1, 2, -1]:
        coord = coords[i]
        target = coords[i + 1]
        dist_x = (target[0] - coord[0]) / edges
        dist_y = (target[1] - coord[1]) / edges
        l = [[coord[0] + dist_x * j, coord[1] + dist_y * j] for j in range(1, edges)]
        polygon.extend(l)
    polygon.append(coords[0])
    return polygon

def check_overlap(polygon1, polygon2, threshold=0.1):
    p1 = make_valid(Polygon(polygon1))
    p2 = make_valid(Polygon(polygon2))
    if p1 is not None and p2 is not None:
        return p1.intersection(p2).area / p1.union(p2).area > threshold
    return False

def process_polygon(poly):
    if poly is None:
        return
    if len(poly) < 3:
        return
    p = make_valid(Polygon(poly))
    if type(p) == MultiPolygon:
        return max(p, key=lambda a: a.area)
    return p

def get_bounding_box(point_list):
    x_list = [point[0] for point in point_list]
    y_list = [point[1] for point in point_list]
    z_list = [point[2] for point in point_list]
    return [[min(x_list), min(y_list), min(z_list)], [max(x_list), max(y_list), max(z_list)]]

def get_nb_common_points(points1, points2):
    """
    Get the number of common points between two lists of 3D points.

    Parameters:
    points1 (list of list of float): First list of 3D points.
    points2 (list of list of float): Second list of 3D points.

    Returns:
    int: Number of common points.
    """
    t = cdist(points1, points2)
    return (t < 0.01).sum()

def get_polyhull(points: np.ndarray):
    np_cloud = sor_filter(points)
    if len(np_cloud) != 0:
        hull = ConvexHull(np_cloud)
        faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=np.int32), hull.simplices)).flatten()
        poly = pv.PolyData(hull.points, faces)
        return poly

def check_contact(ph1, ph2):
    _, x =  ph1.collision(ph2, contact_mode=1)
    return x != 0

def project_points_to_plane(points, plane_origin, plane_normal):
    """Project points to a plane."""
    vec = points - plane_origin
    dist = np.dot(vec, plane_normal)
    return points - np.outer(dist, plane_normal)


def eccentricity_filter(points: np.ndarray,camera_center: np.ndarray, eccentricity_threshold=0.5, k = 1):
    """Filter points based on the circle eccentricity."""
    d = np.linalg.norm(points - camera_center, axis=1)
    d_corr = d - np.min(d)

    d_exc = np.linalg.norm(
        points - np.roll(points, len(points) // 2, axis=0), axis=1
    )
    semi_minor = min(d_exc)
    semi_major = max(d_exc)
    eccentricity = math.sqrt(1 - semi_minor / semi_major)
    if eccentricity > eccentricity_threshold:
        d_max = np.mean(d_corr) + k * np.std(d_corr)
        return points[d_corr < d_max]
    return points


def sor_filter(points: np.ndarray, k1=8, k2=2):
    """
    Applies Statistical Outlier Removal (SOR) filter to a set of points.

    Parameters:
    points (np.ndarray): A numpy array of points to be filtered.
    k1 (int): The number of nearest neighbors to consider for each point. Default is 8.
    k2 (int): The multiplier for the standard deviation to determine the threshold distance. Default is 1.

    Returns:
    np.ndarray: A numpy array of filtered points.
    """
    points = np.array(points)  # Convert the points to a numpy array
    tree = KDTree(points)  # Create a KDTree for efficient neighbor search
    dists = []  # List to store distances of neighbors
    pts_dist = {}  # Dictionary to store mean distance of neighbors for each point

    for i, p in enumerate(points):
        d_kdtree, idx = tree.query(p, k1)  # Query the k1 nearest neighbors
        dists.extend(list(d_kdtree))  # Extend the distances list with the distances of the neighbors
        pts_dist[i] = np.mean(d_kdtree)  # Store the mean distance of the neighbors for the current point

    max_dist = np.mean(dists) + k2 * np.std(dists)  # Calculate the maximum allowable distance
    filtered_points = points[[i for i in pts_dist.keys() if pts_dist[i] < max_dist]]  # Filter points based on the threshold
    return filtered_points  # Return the filtered points


def transform_shapefile(shapefile, transformation, shift=None):

    gdf = gpd.read_file(shapefile)

    transf = pd.read_csv(transformation, sep=" ", header=None)
    a, b, c, d, e, f, g, h, i = transf.to_numpy()[:3, :3].flatten()
    x_off, y_off, z_off = transf.to_numpy()[:3, 3]

    if shift is not None and len(shift) == 2:
        x_off += shift[0]
        y_off += shift[1]

    gdf.geometry = gdf.affine_transform([a, b, c, d, e, f, g, h, i, x_off, y_off, z_off])

    export_filename = os.path.basename(shapefile).split(".")[0] + "_transformed.shp"
    export_path = os.path.join(os.path.dirname(shapefile), export_filename)
    gdf.to_file(export_path)
    return gdf


def fit_plane(point_cloud: np.ndarray):
    """
    input
        point_cloud : list of xyz values　numpy.array
    output
        plane_v : (normal vector of the best fit plane)
        com : center of mass
    """

    com = np.sum(point_cloud, axis=0) / len(point_cloud)
    # calculate the center of mass
    q = point_cloud - com
    # move the com to the origin and translate all the points (use numpy broadcasting)
    Q = np.dot(q.T, q)
    # calculate 3x3 matrix. The inner product returns total sum of 3x3 matrix
    la, vectors = np.linalg.eig(Q)
    # Calculate eigenvalues and eigenvectors
    plane_v = vectors.T[np.argmin(la)]
    # Extract the eigenvector of the minimum eigenvalue

    return plane_v, com


def cut_point_cloud(point_cloud, center: list, distance = 2):
    box = [center[0] - distance, center[0] + distance,
         center[1] - distance, center[1] + distance,
         center[2] - distance, center[2] + distance]
    return point_cloud.clip_box(box, invert=False)











