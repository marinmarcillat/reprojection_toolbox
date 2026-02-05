import pyvista as pv
import pandas as pd
import seaborn as sns
import colorcet as cc

# CM
# mesh = pv.read(r"D:\00_Local_scale\Explorer_canyon\01-Reconstructions\model_offset_EPSG32629.ply")
# annotations = pd.read_csv(r"D:\00_Local_scale\Explorer_canyon\03-Annotations\annotations_EPSG32629shift_wodiameter.csv")
#offset_x = 451000
#offset_y = 5369000
#label_col = "label"

# CG
mesh = pv.read(r"D:\00_Local_scale\coral_garden_22\reconstruction\CG22_EPSG32629shift_aligned_mesh_decimated.ply")
annotations = pd.read_csv(r"D:\00_Local_scale\coral_garden_22\02-reprojection\EPSG32629_aligned_grouped\annotations_EPSG32629shift_aligned_011025.csv")
offset_x = 0
offset_y = 0
label_col = "Classification"


labels = annotations[label_col].unique()
colors = sns.color_palette(cc.glasbey_light, n_colors=len(labels))
label_color_map = {label: colors[i] for i, label in enumerate(labels)}

mesh['z'] = mesh.points[:, 2]

plotter = pv.Plotter()

for label in labels:
    color = label_color_map[label]
    rgb = [int(c * 255) for c in color]
    points = annotations.loc[annotations[label_col] == label][["x", "y", "z"]]
    points['x'] = points['x'] - offset_x
    points['y'] = points['y'] - offset_y
    points = points.to_numpy()
    point_cloud = pv.PointSet(points)
    plotter.add_mesh(point_cloud, color = rgb, render_points_as_spheres = True, point_size = 4, label = label)
plotter.add_mesh(mesh, scalars = 'z')
#plotter.enable_eye_dome_lighting()
plotter.add_legend()
plotter.add_axes()
#plotter.show()

path = plotter.generate_orbital_path(n_points=600, shift=mesh.length)
plotter.open_gif('orbit.gif')
plotter.orbit_on_path(path, write_frames=True)
plotter.close()