import reprojection.reprojection as reprojection
from reprojection.reprojection_database import Camera
import reprojection.metashape_utils as mu
from tqdm import tqdm
from reprojection.geometry import rectangle_to_polygons
import pandas as pd
import os

def camera_reprojector_to_db(session, camera):
    exists = session.query(Camera).filter_by(name=camera.camera.label).first()
    if not exists:
        c = Camera(name=camera.camera.label,
                   abs_path=camera.camera.photo.path,
                   center_x=camera.camera.center[0],
                   center_y=camera.camera.center[1],
                   center_z=camera.camera.center[2],
                   poly_hull_file=camera.contour_file)
        session.add(c)
        session.commit()
    else:
        c = exists
    return c


def chunk_to_camera_reprojector(chunk, db_dir):
    model = mu.get_sparse_model(chunk)

    cph_dir = os.path.join(db_dir, "camera_polyhulls")
    if not os.path.exists(cph_dir):
        os.makedirs(cph_dir)

    print("get camera reprojectors")
    meta_cameras = [camera for camera in chunk.cameras if camera.transform]
    return [
        reprojection.CameraReprojector(
            camera, chunk, model, cph_dir
        )
        for camera in tqdm(meta_cameras)
    ]

def chunk_to_img_labels(chunk):
    cameras = [[os.path.basename(camera.photo.path), int(camera.photo.meta["File/ImageWidth"]), int(camera.photo.meta["File/ImageHeight"])] for camera in chunk.cameras if camera.transform]
    camera_pd = pd.DataFrame(cameras, columns=["filename", "width", "height"])
    camera_pd["points"] = camera_pd.apply(lambda x: rectangle_to_polygons([[0,0],[x.width, 0], [x.width, x.height],[0, x.height]]), axis=1)
    camera_pd["points"] = camera_pd["points"].apply(lambda x: [z for y in x for z in y])
    camera_pd["label_name"] = camera_pd["filename"]
    camera_pd["confidence"] = 1
    return camera_pd[["filename", "label_name", "confidence", "points"]]


