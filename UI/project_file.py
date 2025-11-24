import json, os
from PyQt5.QtWidgets import QDialog, QFileDialog
from UI.project_window import Ui_Dialog

project_template = {
    'name': '',
    'project_directory': '',
    'video_directory': '',
    'image_directory': "",
    "metashape_project_path": "",
    "reprojection_image_directory": "",
    'inference_model_path': '',
    'annotation_report_path': ''
}


def read_json(file_path):
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path: The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary, or None if an error occurs.
    """
    #try:
    with open(file_path, "r") as f:
        return json.load(f)
    #except Exception:
    #    print("Error reading, please check your project json file and try again.")
    #    return None


def write_json(file_path, project_config):
    """
    Writes a dictionary to a JSON file.

    Args:
        file_path: The path to the JSON file.
        project_config: The dictionary to be written.

    Returns:
        None.
    """
    with open(file_path, "w") as outfile:
        json.dump(project_config, outfile, indent=4)


class NewProjectDialog(QDialog, Ui_Dialog):
    global inputs
    def __init__(self, project_name, directory):
        super().__init__()
        self.setupUi(self)
        self.proj_name = project_name
        self.proj_dir = directory

        self.proj_file = os.path.join(directory, project_name + ".rpj")

        self.img_dir_B.clicked.connect(lambda: self.select_dir(self.img_dir))
        self.mtshp_prj_B.clicked.connect(lambda: self.select_file(self.mtshp_prj, "*.psx"))
        self.img_inf_dir_B.clicked.connect(lambda: self.select_dir(self.img_inf_dir))
        self.inf_model_B.clicked.connect(lambda: self.select_file(self.inf_model, "*.pt"))
        self.report_path_B.clicked.connect(lambda: self.select_file(self.report_path, "*.csv"))


        self.buttonBox.accepted.connect(self.launch)

    def select_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(None, 'Open Directory', r"")
        if dir_path:
            line_edit.setText(dir_path)

    def select_file(self, line_edit, file_filter):
        options = QFileDialog.Options()
        file_path = QFileDialog.getOpenFileName(self, "Select File", "", file_filter, options=options)
        if file_path:
            line_edit.setText(file_path[0])



    def launch(self):
        project_config = project_template.copy()
        project_config['name'] = self.proj_name
        project_config['project_directory'] = self.proj_dir

        project_config['image_directory'] = self.img_dir.text()
        project_config['metashape_project_path'] = self.mtshp_prj.text()
        project_config["reprojection_image_directory"] = self.img_inf_dir.text()
        project_config['inference_model_path'] = self.inf_model.text()
        project_config['annotation_report_path'] = self.report_path.text()
        write_json(self.proj_file, project_config)