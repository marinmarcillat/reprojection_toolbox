import pyvista as pv

grid = pv.ImageData(r"D:\data_coral_garden\raw\other\LampaulCanyon_10m_renorm_elevation.tif")
subset = grid.threshold(value=0, invert = True)
terrain = subset.warp_by_scalar(factor = 0.2)