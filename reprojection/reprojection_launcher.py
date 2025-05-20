import os
import sys
import pandas as pd
from PyQt5 import QtCore, QtGui

import reprojection.annotations as annotations
import reprojection.camera_utils as cu
import reprojection.reprojection_database as rdb

import reprojection.metashape_utils as mu




class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    encoding = 'utf-8'

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class ReprojectionThread(QtCore.QThread):
    prog_val = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, gui, chunk, db_dir):
        super(ReprojectionThread, self).__init__()
        self.running = True
        self.gui = gui
        self.chunk = chunk
        self.db_dir = db_dir

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
        print("Create camera reprojector objects")
        cameras_reprojectors = cu.chunk_to_camera_reprojector(self.chunk, self.db_dir)
        self.prog_val.emit(10)

        if (self.gui.resetCb.isChecked()) or not (self.gui.project_config.get("rp_db", False)):
            print("Get all tie points")
            tie_points = mu.get_all_tie_points(self.chunk, self.db_dir)
            self.prog_val.emit(20)

            print("Creating DB and adding annotations and reprojections")
            if self.gui.project_config.get("annotation_report_path", False):
                inference_report = pd.read_csv(self.gui.project_config["annotation_report_path"])
                try:
                    inference_report = inference_report[["filename", "label_name", "confidence", "points"]]
                except KeyError:
                    print("Detected biigle format, adding confidence column")
                    inference_report = inference_report[["filename", "label_name", "points"]].assign(confidence=1)

                if self.gui.img_labels_cb.isChecked():
                    img_label = cu.chunk_to_img_labels(self.chunk)
                    inference_report = pd.concat([inference_report, img_label], axis=0)
                    inference_report.index = pd.RangeIndex(len(inference_report.index))
            else:
                inference_report = cu.chunk_to_img_labels(self.chunk)

            session = annotations.inference_report_to_reprojection_database(self.db_dir, inference_report,
                                                                                 cameras_reprojectors, tie_points)

        else:
            print("Getting DB from existing")
            session, _ = rdb.open_reprojection_database_session(self.db_dir, False)

        self.prog_val.emit(50)

        print("Adding annotations to individuals")
        session = annotations.annotations_to_individuals(session, self.db_dir)
        self.prog_val.emit(80)
        print("Plotting reprojections on metashape")
        mu.plot_all_annotations(session, cameras_reprojectors, self.chunk, self.gui.project_config['name'], pointify = self.gui.pointify.isChecked(), confidence_threshold = 0.5)
        print("Reprojection finished")
        self.prog_val.emit(100)

        self.finished.emit()
        self.running = False



