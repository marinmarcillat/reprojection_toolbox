import Metashape
import os
from reconstruction.reconstruct import post_reconstruction


def create_meta_project():
    doc = Metashape.Document()

    doc.save('project.psx')
    doc.save()
    return doc

def open_meta_project(path):
    doc = Metashape.Document()
    doc.open(path)
    return doc

def import_project(qt, path):
    qt.project_config["metashape"]["project"]  = path
    doc = open_meta_project(path)
    if doc.chunk is not None:
        chunk = doc.chunk
        if len(chunk.cameras) != 0:
            c = chunk.cameras[0]
            img_dir = os.path.dirname(c.photo.path)
            qt.project_config["image_dir"] = img_dir

        if chunk.model and chunk.point_cloud:
            post_reconstruction(qt, doc)

    return doc

def find_files(folder, types):
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

def add_images(chunk, image_dir):

    photos = find_files(image_dir, [".jpg", ".jpeg", ".tif", ".tiff"])
    chunk.addPhotos(photos)

    chunk.analyzeImages()
    for c in chunk.cameras:
        if float(c.meta['Image/Quality']) < 0.5:
            print(f"Removed image {c.label} with score {c.meta['Image/Quality']}")
            c.enabled = False



def add_navigation(doc, navigation_file):
    crs = Metashape.CoordinateSystem("EPSG::4326")
    doc.chunk.importReference(path=navigation_file, columns='nxyz', crs=crs, delimiter=',')
    doc.save()




