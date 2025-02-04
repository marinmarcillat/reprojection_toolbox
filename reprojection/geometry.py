import math
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from scipy.spatial import ConvexHull
import pyvista as pv
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
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


def excentricity_filter(points: np.ndarray,camera_center: np.ndarray, excentricity_threshold=5, k = 0.1):
    """Filter points based on their excentricity."""
    d = np.linalg.norm(points - camera_center, axis=1)
    d_corr = d - np.min(d)
    excentricity = np.quantile(d_corr, 0.75) / np.quantile(d_corr, 0.25)
    if excentricity > excentricity_threshold:
        d_max = np.mean(d_corr) + k * np.std(d_corr)
        r = points[d_corr < d_max]
        return r
    return points


def sor_filter(points: np.ndarray, k1=8, k2=1):
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
