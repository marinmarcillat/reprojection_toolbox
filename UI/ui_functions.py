import os
import reprojection.reprojection_database as rdb
from UI.get_meta_status import get_meta_status
from PyQt5.QtWidgets import QMessageBox

def reset_status_ui(qt):
    """
    Reset the status bar to its default value.
    """

    qt.projectPath.setText("")
    qt.metaProject.setText("")
    qt.metaChunk.clear()
    qt.imageDir.setText("")
    qt.imageNb.setText("")
    qt.inferenceModelPath.setText("")
    qt.overlappingImageDir.setText("")
    label_list = [
        qt.tiePoints,
        qt.lowRes,
        qt.highRes,
        qt.reprojDB,
        qt.reprojected,
        qt.individual
    ]
    for label in label_list:
        label.setStyleSheet("QLabel {color : black; font-weight: roman}")

    action_list = [
        qt.reconstruct,
        qt.overlapping,
        qt.inference,
        qt.reprojection
    ]
    for action in action_list:
        action.setEnabled(False)

    checks_list = [
    "read_only",
    "overlapping_images",
    "tie_points",
    "aligned",
    "lowRes",
    "highRes",
    "rp_db",
    "reprojected",
    "individual",
    ]
    for check in checks_list:
        qt.project_config[check] = False

    qt.progressBar.setValue(0)

def get_status(qt):
    chunk_name = qt.project_config["chunk_name"]

    if qt.project_config is None:
        print("No project loaded")
        return 0

    reset_status_ui(qt)

    if not os.path.exists(qt.project_config["project_directory"]):
        print("No project path or non existent")
        return 0

    qt.projectPath.setText(qt.project_config["project_directory"])

    if not os.path.exists(qt.project_config["image_directory"]):
        print("No image directory or non existent")
        return 0

    qt.imageDir.setText(qt.project_config["image_directory"])
    qt.imageNb.setText(str(len(os.listdir(qt.project_config["image_directory"]))))


    if not os.path.exists(qt.project_config["metashape_project_path"]):
        print("Metashape project not found")
    else:
        qt.metaProject.setText(qt.project_config["metashape_project_path"])

        meta_status = get_meta_status(qt, chunk_name)
        print(f'Metashape status: {meta_status}')

    if qt.project_config.get("read_only", False) :
        read_only_warning()


    if os.path.exists(qt.project_config["inference_model_path"]):
        qt.inferenceModelPath.setText(qt.project_config["inference_model_path"])

    overlapping_images_dir = os.path.join(qt.project_config["project_directory"], "overlapping_images")
    if os.path.exists(overlapping_images_dir):
        if len(os.listdir(overlapping_images_dir)) != 0:
            qt.overlappingImageDir.setText(overlapping_images_dir)
            qt.project_config["overlapping_images"] = True
        else:
            qt.project_config["overlapping_images"] = False
    else:
        qt.project_config["overlapping_images"] = False

    db_path = os.path.join(qt.project_config["project_directory"], f"reprojection_{qt.project_config['name']}", "reprojection.db")
    if os.path.exists(db_path):
        session_status_manager(qt)

    if (os.path.exists(qt.project_config["image_directory"])
            and os.path.exists(qt.project_config["project_directory"])):
        qt.reconstruct.setEnabled(True)

    if (
        qt.project_config.get("aligned", False)
        and not qt.project_config["overlapping_images"]
    ):
        qt.overlapping.setEnabled(True)

    if qt.project_config.get("lowRes", False):
        qt.reconstruct.setEnabled(False)

    if (qt.project_config.get("overlapping_images", False)
            and os.path.exists(qt.project_config["inference_model_path"])):
        qt.inference.setEnabled(True)

    if  qt.project_config.get("lowRes", False):
        qt.reprojection.setEnabled(True)


def session_status_manager(qt):
    qt.project_config["rp_db"] = True
    qt.reprojDB.setStyleSheet("QLabel {color : green; font-weight: bold}")
    print("Reprojection database found")
    if qt.session is not None:
        rdb.get_db_status(qt, qt.session)
    else:
        session, _ = rdb.open_reprojection_database_session(os.path.join(qt.project_config["project_directory"], f"reprojection_{qt.project_config['name']}"), False)
        rdb.get_db_status(qt, session)
        session.close()

    if qt.project_config.get("reprojected", False):
        qt.reprojected.setStyleSheet("QLabel {color : green; font-weight: bold}")
    if qt.project_config.get("individual", False):
        qt.individual.setStyleSheet("QLabel {color : green; font-weight: bold}")




def read_only_warning():
    msg_box_name = QMessageBox()
    msg_box_name.setIcon(QMessageBox.Warning)
    msg_box_name.setWindowTitle("Read-only project")
    msg_box_name.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg_box_name.setText("The Metashape project is read-only. Close the app?")
    retval = msg_box_name.exec_()
    if retval == 1024:
        print("closing...")
        quit()
    else:
        print("Ignoring read only status")
        msg_box_name.close()


