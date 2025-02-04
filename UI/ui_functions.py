import os
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
        qt.runAll,
        qt.reconstruct,
        qt.overlapping,
        qt.inference,
        qt.reprojection
    ]
    for action in action_list:
        action.setEnabled(False)

    checks_list = [
    "overlapping_images",
    "tie_points",
    "aligned",
    "lowRes",
    "highRes"
    ]
    for check in checks_list:
        qt.project_config[check] = False

    qt.progressBar.setValue(0)

def get_status(qt):

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

        meta_status = get_meta_status(qt)
        print(f'Metashape status: {meta_status}')
        if meta_status == "read_only":
            read_only_warning()


    if os.path.exists(qt.project_config["inference_model_path"]):
        qt.inferenceModelPath.setText(qt.project_config["inference_model_path"])

    overlapping_images_dir = os.path.join(qt.project_config["project_directory"], "overlapping_images")
    if not os.path.exists(overlapping_images_dir):
        os.makedirs(overlapping_images_dir)

    if len(os.listdir(overlapping_images_dir)) != 0:
        qt.overlappingImageDir.setText(overlapping_images_dir)
        qt.project_config["overlapping_images"] = True
        qt.project_config["reprojection_image_directory"] = overlapping_images_dir

    if (os.path.exists(qt.project_config["image_directory"])
            and os.path.exists(qt.project_config["project_directory"])):
        qt.reconstruct.setEnabled(True)

    if qt.project_config.get("aligned", False) and not (len(os.listdir(overlapping_images_dir)) != 0):
        qt.overlapping.setEnabled(True)

    if qt.project_config.get("lowRes", False):
        qt.reconstruct.setEnabled(False)

    if (qt.project_config.get("overlapping_images", False)
            and os.path.exists(qt.project_config["inference_model_path"])):
        qt.inference.setEnabled(True)

    if (qt.project_config.get("annotation_report_path", False)
            and os.path.exists(qt.project_config["reprojection_image_directory"])
            and qt.project_config.get("lowRes", False)):
        qt.reprojection.setEnabled(True)

    if (os.path.exists(qt.project_config["image_directory"])
            and os.path.exists(qt.project_config["project_directory"])
            and os.path.exists(qt.project_config["inference_model_path"])):
        qt.runAll.setEnabled(True)




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


