import os.path
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog)
from PyQt5 import QtCore, QtGui
import logging
import Metashape
import pandas as pd

import UI.project_file as project_file
from UI.main_window import Ui_MainWindow
import UI.ui_functions as ui_functions
import contextlib
from reconstruction import reconstruct
from reprojection import metashape_utils as mu
from object_detection import fifty_one_utils as fou
from object_detection import sahi_inference as sahi
from object_detection import inference_launcher
from reprojection import camera_utils as cu
from reprojection import annotations
from reprojection import reprojection_database as rdb
from reprojection import reprojection_launcher



class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    encoding = 'utf-8'

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignals()

        self.project_config = None
        self.project_config_path = None

        self.log_path = None

        self.doc = None
        self.chunk = None
        
        self.session = None

        self.processing = False

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)


    def connectSignals(self):
        self.actionNew.triggered.connect(self.new_project)
        self.actionOpen.triggered.connect(self.open_project)
        self.reload.clicked.connect(lambda: ui_functions.get_status(self))
        self.reconstruct.clicked.connect(self.launch_reconstruction)
        self.overlapping.clicked.connect(self.launch_overlapping)
        self.inference.clicked.connect(self.launch_inference)
        self.reprojection.clicked.connect(self.launch_reprojection)

    def new_project(self):
        print("create new project (not yet implemented)")

    def open_project(self):
        options = QFileDialog.Options()
        file_path = QFileDialog.getOpenFileName(self, "Open project file", "", "*.rpj", options=options)
        self.project_config = project_file.read_json(file_path[0])
        self.project_config_path = file_path[0]
        self.log_path = os.path.join(self.project_config['project_directory'], self.project_config['name'] + ".log")
        ui_functions.get_status(self)

    def get_meta_chunk(self):
        if self.doc is None:
            try:
                self.doc = Metashape.Document()
                self.doc.open(self.project_config['metashape_project_path'])
            except:
                print("Error opening project")
                return None

        try:
            chunk_id = int(self.metaChunk.currentIndex())
        except ValueError:
            chunk_id = 0

        self.chunk = self.doc.chunks[chunk_id]
        return 1


    def launch_reconstruction(self):

        print("launching reconstruction")
        if not os.path.exists(self.project_config['metashape_project_path']):
            self.doc = Metashape.Document()
            metashape_project_path = os.path.join(self.project_config['project_directory'], self.project_config['name'] + ".psx")
            self.project_config['metashape_project_path'] = metashape_project_path
            self.doc.save(metashape_project_path)
            self.doc.save()
            self.doc.addChunk()
            self.doc.save()
            self.chunk = self.doc.chunk
        elif self.get_meta_chunk() is None:
            return None

        if len(self.chunk.cameras) == 0:
            print("No cameras found, adding those from image directory")
            mu.add_images(self.chunk, self.project_config['image_directory'])
            self.doc.save()

        meta_reconstruction_thread = reconstruct.MetaReconstructionThread(self, self.doc, self.chunk)
        meta_reconstruction_thread.prog_val.connect(self.set_prog)
        meta_reconstruction_thread.finished.connect(self.after_operation)
        meta_reconstruction_thread.start()
        print("launch reconstruction")

    def launch_overlapping(self):
        if self.get_meta_chunk() is None:
            return None


        export_dir = os.path.join(self.project_config["project_directory"], "overlapping_images")
        temp_chunk = mu.get_overlapping_images(self.chunk, export_dir)
        self.doc.remove([temp_chunk])
        self.after_operation()

    def launch_inference(self):
        inference_thread = inference_launcher.InferenceThread(self)
        inference_thread.prog_val.connect(self.set_prog)
        inference_thread.finished.connect(self.after_operation)
        inference_thread.start()
        print("Launch inference")

    def launch_reprojection(self):

        db_dir = os.path.join(self.project_config["project_directory"], f"reprojection_{self.project_config['name']}")
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        if self.get_meta_chunk() is None :
            return None

        if not self.img_labels_cb.isChecked() and  not self.project_config.get("annotation_report_path", False):
            print("No annotation report available. Abort")

        reprojection_thread = reprojection_launcher.ReprojectionThread(self, self.chunk, db_dir)
        reprojection_thread.prog_val.connect(self.set_prog)
        reprojection_thread.finished.connect(self.after_reprojection)
        reprojection_thread.start()
        print("launch reconstruction")



    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        cursor = self.debug.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.debug.setTextCursor(cursor)
        self.debug.ensureCursorVisible()

        if self.log_path is not None:
            with open(self.log_path, 'a', encoding="utf-8") as f:
                f.write(text)

    def set_prog(self, val):
        self.progressBar.setValue(val)

    def after_reprojection(self):
        self.doc.save()
        self.after_operation()

    def after_operation(self):
        print("Operation finished")
        ui_functions.get_status(self)
        project_file.write_json(self.project_config_path, self.project_config)
        self.processing = False

    def closeEvent(self, event):
        del self.doc
        with contextlib.suppress(AttributeError):
            self.session.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())