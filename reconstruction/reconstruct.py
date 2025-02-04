import Metashape
import zipfile
from PyQt5 import QtCore, QtGui
from shutil import copy
import os
import sys


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    encoding = 'utf-8'

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class MetaReconstructionThread(QtCore.QThread):
    prog_val = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, gui, doc, chunk):
        super(MetaReconstructionThread, self).__init__()
        self.running = True
        self.doc = doc
        self.gui = gui
        self.chunk = chunk

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
        reconstruct_model_metashape(self.doc, self.chunk)
        self.finished.emit()
        self.running = False




def reconstruct_model_metashape(doc, chunk):
    print('matching photos')
    chunk.matchPhotos(keypoint_limit=40000, tiepoint_limit=10000, generic_preselection=True,
                      reference_preselection=True)
    doc.save()

    print('aligning cameras')
    chunk.alignCameras()
    doc.save()

    print('building low res model')
    chunk.buildModel(source_data=Metashape.DataSource.TiePointsData)
    doc.save()

    print('building depth maps')
    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)
    doc.save()

    print('building high res model')
    chunk.buildModel(source_data=Metashape.DepthMapsData)
    doc.save()

    print('building uv')
    chunk.buildUV(page_count=2, texture_size=4096)
    doc.save()

    print('building texture')
    chunk.buildTexture(texture_size=4096, ghosting_filter=True)
    doc.save()

    print('building point cloud')
    chunk.buildPointCloud()
    doc.save()




def post_reconstruction(qt, doc):
    export = os.path.join(qt.project_config['project_directory'], 'export')
    if not os.path.exists(export):
        os.makedirs(export)

    model_obj_path = os.path.join("export","model.obj")
    doc.chunk.exportModel(path=model_obj_path, format = Metashape.ModelFormatOBJ,
                      texture_format=Metashape.ImageFormat.ImageFormatJPEG, save_texture=True)

    model_ply_path = os.path.join("export","model.ply")
    doc.chunk.exportModel(path= model_ply_path, format = Metashape.ModelFormatPLY, binary = True,
                      save_colors = True)

    ref_path = os.path.join("export", "reference.csv")
    doc.chunk.exportReference(path=ref_path, columns = "nuvwdef", format = Metashape.ReferenceFormatCSV,
                              items = Metashape.ReferenceItemsCameras)

    qt.project_config['metashape']['3D_model'] = {
        'model_ply_path': model_ply_path,
        'model_obj_path': model_obj_path,
        'ref_path': ref_path
    }

    point_path = os.path.join("export", "point_cloud.pcd")
    doc.chunk.exportPointCloud(path=point_path, format=Metashape.PointCloudFormatPCD,
                          save_colors=True)
    qt.project_config["metashape"]["point_cloud"] = point_path










