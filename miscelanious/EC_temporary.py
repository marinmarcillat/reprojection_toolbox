import Metashape
import numpy as np
import pandas as pd


project_path = r"F:\these\Explorer_NOC\Explorer_Bare.psx"

doc = Metashape.Document()
doc.open(project_path)
chunk = doc.chunk

annotations = []
for shape in chunk.shapes:
    label = shape.label
    if shape.geometry.type == Metashape.Geometry.Type.PointType: # Points
        point = list(np.array(shape.geometry.coordinates[0]))
        diameter = -1
    else:  # Polygons
        points = np.array([list(coord) for coord in shape.geometry.coordinates])
        center = points.mean(axis=1)
        perimeter_length = shape.perimeter3D()
        diameter = perimeter_length / np.pi
        point = list(center[0])
    point.insert(0, label)
    point.append(diameter)
    annotations.append(point)

annotations_df = pd.DataFrame(annotations, columns=["label", "x", "y", "z", "diameter"])
annotations_df.to_csv(r"F:\these\Explorer_NOC\annotations\Marin_annotations.csv", index=False)

