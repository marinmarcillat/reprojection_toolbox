import pyvista as pv
import numpy as np
import geopandas as gpd
from pyvista import Plotter
from shapely.geometry import Point
from scipy.spatial import cKDTree as KDTree
from collections import Counter
import pickle
from tqdm import tqdm



def process_labels(annotations_data, label_map, to_delete):
    annotations_data.replace(label_map, inplace=True)
    annotations_data = annotations_data[~annotations_data['NAME'].isin(to_delete)].copy()
    annotations_data.reset_index(drop=True, inplace=True)
    return annotations_data

def get_abundances(annotations_data):
    labels = annotations_data['NAME'].unique().tolist()
    for label in labels:
        data = annotations_data[annotations_data['NAME'] == label]
        print(f"{label}: {len(data)}")


def get_nearest(df, threshold):
    points_list = np.array([
        [float(row.geometry.x), float(row.geometry.y), float(row.geometry.z)]
        for _, row in df.iterrows()
    ])
    tree = KDTree(points_list)
    return list(tree.query_pairs(r=threshold))

def gpd2pv(df, offset = [0,0]):
    points_list = np.array([
        [float(row.geometry.x) - offset[0], float(row.geometry.y) - offset[1], float(row.geometry.z)]
        for _, row in df.iterrows()
    ])
    return pv.PolyData(points_list)

def remove_closest_duplicates(annotations_data, label, threshold):
    data = annotations_data[annotations_data['NAME'] == label].copy()
    data.reset_index(drop=True, inplace=True)
    pairs = get_nearest(data, threshold)

    while len(pairs) > 1:
        print(len(pairs), end='\x1b[1K\r')
        idx1, idx2 = pairs[0]

        points = data.iloc[[int(idx1), int(idx2)]].geometry.values
        coords = np.array([[pt.x, pt.y, pt.z] for pt in points])
        mid_point =  Point(coords.mean(axis=0))


        global_idx1 = annotations_data[annotations_data['FID'] == data.iloc[int(idx1)]['FID']].index[0]
        global_idx2 = annotations_data[annotations_data['FID'] == data.iloc[int(idx2)]['FID']].index[0]

        # Update geometries
        annotations_data.at[global_idx1, 'geometry'] = mid_point
        annotations_data.drop(global_idx2, inplace=True)
        data.drop(idx2, inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Recalculate nearest neighbors
        pairs = get_nearest(data, threshold)
    return annotations_data


if __name__ == "__main__":
    shp_file = r"D:\00_Local_scale\vertical_wall_22\02-reprojection\reprojections_annotations_700_EPSG32629_230925.shp"
    model = pv.read(
        r"D:\00_Local_scale\vertical_wall_22\03-reconstruction\01_Metashape_output\VW_22_reproj_colcorRGB_EPSG32629shift.ply")
    labels_to_refine = [["SM394 Araeosoma fenestratum", 0.05], ["SM92 Actiniaria msp1", 0.0564]]

    to_delete = [
        "Save for later",
        "Incertae sedis",
        "Unlisted",
        "Unknown species",
        "Gadiformes",
        "SM1133/SM1143 Madrepora oculata or Desmophyllum pertusum (indistinguishable)",
        "SM1133 Madrepora oculata1",
        "Fragment",
        "Argentiniformes",
        "SM440 Trachyscorpia cristulata",
        "Actiniaria",
        "Ceriantharia",
        "Tunicata",
        "Antipatharia",
        "Asteroidea",
        "Leiopathidae",
        "Echinoidea",
        "Echinidae",
        "Pennatulacea",
        "SM305 Pseudarchaster",
        "Chirostylids",
        "SM700 Plexauridae msp6",
        "Cidaroida",
        "Alcyonacea",
        "SM698 Chrysogorgiidae msp.1",
        "Crinoidea",
        "Primnoidae",
        "Whip"
    ]

    label_map = {
        "SM235 Bathynectes longispina": "Crabs",
        "SM206 Chaceon msp1": "Crabs",
        "SM236 Paromola cuvieri": "Crabs",
        "True crabs": "Crabs",
        "Spider crabs": "Crabs",
        "Caryophylliidae": "Solitary",
        "Stichopathes sp. (undefined)": "Stichopathes",
        "SM130 Stichopathes cf. gravieri": "Stichopathes",
        "Crust-like": "Porifera",
        "Cup-like": "Porifera",
        "Massive": "Porifera",
        "SM139 Bathypathes  msp.2": "Schizopathidae",
        "SM170 Bathypathes  msp.1": "Schizopathidae",
        "SM134 Parantipathes  msp.2": "Schizopathidae",
        "SM154 Parantipathes  msp.1": "Schizopathidae",
        "SM181 Parantipathes hirondelle": "Schizopathidae",
        "SM288 Porania (Porania) pulvillus (orange morph)": "Asteroidea cushion body form",
        "SM302 Porania (Porania) pulvillus (purple morph)": "Asteroidea cushion body form",
        "Stylasteridae": "Hydrozoa"
    }

    annotations_data = gpd.read_file(shp_file)

    annotations_data = process_labels(annotations_data, label_map, to_delete)

    for label, threshold in labels_to_refine:
        annotations_data = remove_closest_duplicates(annotations_data, label, threshold)

    for label in annotations_data['NAME'].unique():
        annotations_data = remove_closest_duplicates(annotations_data, label, 0.02)

    nearest = get_nearest(annotations_data, 0.1)
    print(f"Found {len(nearest)} pairs of points closer than 10 cm")
    res = []
    for idx1, idx2 in nearest:
        label_1, label_2 = annotations_data.iloc[int(idx1)]['NAME'], annotations_data.iloc[int(idx2)]['NAME']
        res.append((label_1, label_2))

    pair_counts = Counter(res)
    most_common_pairs = pair_counts.most_common(10)
    print("Most frequent label pairs:")
    pair_selection = []
    for pair, count in most_common_pairs:
        print(f"{pair}: {count}")
        if [pair[1], pair[0]] not in pair_selection and [pair[0], pair[1]] not in pair_selection:
            pair_selection.append(pair)


    res = {}
    for pair in pair_selection:
        pair_cloud = []
        for idx1, idx2 in nearest:
            label_1, label_2 = annotations_data.iloc[int(idx1)]['NAME'], annotations_data.iloc[int(idx2)]['NAME']
            if (label_1, label_2) == pair or (label_2, label_1) == pair:
                print(f"Pair {pair} found at indices {idx1}, {idx2}")
                points = annotations_data.iloc[[int(idx1), int(idx2)]].geometry.values
                coords = np.array([[pt.x - 609000, pt.y - 5274000, pt.z] for pt in points])
                mid_point =  coords.mean(axis=0)
                pair_cloud.append(mid_point)
        res[str(pair)] = pv.PolyData(pair_cloud)


    with open('data.pickle', 'wb') as f:
        pickle.dump(res, f)

    plotter = Plotter()
    for label, pcd in res.items():
        plotter.add_points(pcd, color=np.random.rand(3), point_size=8, render_points_as_spheres=True, label=label)
    plotter.add_mesh(model, scalars='RGB', rgb=True)
    plotter.show_grid()
    plotter.enable_eye_dome_lighting()
    plotter.add_legend(bcolor='w')
    plotter.show()

    print("ok")
    print("stop")








"""
res = {}
for label in labels:
    data = annotations_data[annotations_data['NAME'] == label]
    res[label] = gpd2pv(data, offset=[609000, 5274000])



plotter = Plotter()
for label, pcd in res.items():
    if label in ["SM1143 Desmophyllum pertusum", "SM1133 Madrepora oculata"]:
        plotter.add_points(pcd, color=np.random.rand(3), point_size=5, render_points_as_spheres=True, label=label)
plotter.add_mesh(model, scalars='RGB', rgb=True)
plotter.show_grid()
plotter.enable_eye_dome_lighting()
plotter.add_legend(bcolor='w')
plotter.show()"""







