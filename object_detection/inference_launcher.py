from ultralytics import YOLO
from PyQt5 import QtCore, QtGui
import sys
import os
import pandas as pd

from object_detection import fifty_one_utils as fou
from object_detection import inference
from object_detection import sahi_inference as sahi


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    encoding = 'utf-8'

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class InferenceThread(QtCore.QThread):
    prog_val = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, gui):
        super(InferenceThread, self).__init__()
        self.running = True
        self.gui = gui
        self.model_path = self.gui.project_config["inference_model_path"]

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        with open(self.gui.log_path, 'a', encoding="utf-8") as f:
            f.write(text)
        cursor = self.gui.debug.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.gui.debug.setTextCursor(cursor)
        self.gui.debug.ensureCursorVisible()

    def run(self):
        print("create fiftyone dataset")
        export_dir = os.path.join(self.gui.project_config["project_directory"], "overlapping_images")
        dataset = fou.import_image_directory(export_dir, "overlapping_image_dataset")
        dataset.compute_metadata()
        first_img_size = [dataset.first()["metadata"]["width"], dataset.first()["metadata"]["height"]]

        model = YOLO(self.model_path)
        img_sz = model.model.args['imgsz']

        self.prog_val.emit(10)

        if first_img_size[0] > 2*img_sz or first_img_size[1] > 2*img_sz:
            print("Image size is 2x larger than the model input size, going with SAHI")
            dataset_inf = sahi.sahi_inference(self.model_path, dataset, slice=img_sz)
        else:
            print("Image size is smaller than 2x the model input size, going with classic YOLO inference")
            dataset_inf = inference.YOLO_inference(dataset, self.model_path)

        self.prog_val.emit(80)

        print("export inference report")
        inference_report = fou.fo_to_csv(dataset_inf)
        inference_report_path = os.path.join(self.gui.project_config["project_directory"], "inference_report.csv")
        pd.DataFrame(inference_report).to_csv(inference_report_path, index=False)

        self.gui.project_config["annotation_report_path"] = inference_report_path
        self.prog_val.emit(100)

        self.finished.emit()
        self.running = False

if __name__ == "__main__":
    import os
    import fiftyone as fo
    from Biigle.biigle import Api
    from annotation_conversion_toolbox.biigle_dataset import BiigleDatasetExporter

    api = Api()

    volume_id = 684
    label_tree_id = 60
    biigle_image_dir = r"Z:\images\chereef_2022\CG_disjoint"

    model_path = r"D:\02_object_detection\trained_models\AS_CG_yolo11_030925\train\weights\best.pt"
    model = YOLO(model_path)
    img_sz = model.model.args['imgsz']

    img_volume = list(api.get(f"volumes/{volume_id}/filenames").json().values())

    dataset = fou.import_image_directory(biigle_image_dir, "Coral_CG22_0804")

    to_delete = []
    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        if filename not in img_volume:
            to_delete.append(sample.id)
    dataset.delete_samples(to_delete)

    dataset.compute_metadata()
    first_img_size = [dataset.first()["metadata"]["width"], dataset.first()["metadata"]["height"]]

    dataset_inf = sahi.sahi_inference(model_path, dataset, slice=img_sz)

    print("export inference report")
    inference_report = fou.fo_to_csv(dataset_inf)
    inference_report_path = os.path.join(r"D:\04-Community_temporal\02_pretrained_inference", "inference_report_22.csv")
    pd.DataFrame(inference_report).to_csv(inference_report_path, index=False)

    exporter = BiigleDatasetExporter(api=api, volume_id=volume_id, label_tree_id=label_tree_id, biigle_image_dir=biigle_image_dir)
    dataset_inf.export(dataset_exporter=exporter)

    session = fo.launch_app(dataset_inf)
    session.wait()