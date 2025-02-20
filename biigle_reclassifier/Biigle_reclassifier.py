import os
import sys
import time

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog, QApplication, QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5 import QtGui

from biigle_reclassifier.choose_label import SelectWindow
from biigle_reclassifier import utils
from biigle_reclassifier.biigle_reclassifier_ui import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # setting  the geometry of window
        self.setGeometry(100, 100, 1600, 1200)
        #self.showMaximized()

        self.connectActions()


        self.reclass_type.addItems(['Largo', 'Label'])

        self.labels = {}
        self.prev_timer = time.time()
        self.cat = "Largo"
        self.img_id = 0
        self.api = None
        self.timelist = []
        self.img_list = []

        self.lt_id = 0
        self.vol_id = 0

        self.shortcut1 = QShortcut(QKeySequence("Ctrl+1"), self)
        self.shortcut2 = QShortcut(QKeySequence("Ctrl+2"), self)
        self.shortcut3 = QShortcut(QKeySequence("Ctrl+3"), self)
        self.shortcut4 = QShortcut(QKeySequence("Ctrl+4"), self)
        self.shortcut1.activated.connect(self.set_lab1.click)
        self.shortcut2.activated.connect(self.set_lab2.click)
        self.shortcut3.activated.connect(self.set_lab3.click)
        self.shortcut4.activated.connect(self.set_lab4.click)


    def connectActions(self):

        # Reclassifier
        self.work_dir_b.clicked.connect(lambda: self.selectDir(self.work_dir))
        self.launch.clicked.connect(self.start_class)
        self.api_connect.clicked.connect(self.on_connect)
        self.origin_b.clicked.connect(lambda: self.select_label(self.origin))
        self.lab1_b.clicked.connect(lambda: self.select_label(self.lab1))
        self.lab2_b.clicked.connect(lambda: self.select_label(self.lab2))
        self.lab3_b.clicked.connect(lambda: self.select_label(self.lab3))
        self.lab4_b.clicked.connect(lambda: self.select_label(self.lab4))

        self.set_lab1.clicked.connect(lambda: self.nextimage(self.lab1))
        self.set_lab2.clicked.connect(lambda: self.nextimage(self.lab2))
        self.set_lab3.clicked.connect(lambda: self.nextimage(self.lab3))
        self.set_lab4.clicked.connect(lambda: self.nextimage(self.lab4))

    def selectDir(self, line):
        dir_path = QFileDialog.getExistingDirectory(None, 'Open Dir', r"")
        if dir_path:
            line.setText(dir_path)

    def select_label(self, txt):
        sel = SelectWindow(self.labels)
        res = sel.exec_()
        if res == QDialog.Accepted:
            id, value = sel.get_value()
            txt.setText(str(id + "_" + str(value)))

    def on_connect(self):
        try:
            self.lt_id = int(self.label_tree_id.text())
            self.vol_id = int(self.volume_id.text())
        except ValueError:
            print("Invalid ID")
            return

        self.api = utils.connect(self.email.text(), self.token.text())

        if self.api:
            self.api_connect.setText("Connected")
            self.labels = self.api.get(
                f'label-trees/{self.lt_id}'
            ).json()["labels"]
            self.lab1_b.setEnabled(True)
            self.lab2_b.setEnabled(True)
            self.lab3_b.setEnabled(True)
            self.lab4_b.setEnabled(True)
            self.origin_b.setEnabled(True)

    def start_class(self):
        if self.work_dir.text() == "" or self.lab1.text() == "" or self.lab2.text() == "" or self.api == 0:
            print("Missing inputs")
            return

        self.prev_timer = time.time()

        label = self.origin.text().split("_")[0]

        self.cat = self.reclass_type.currentText()

        if self.dwnld.checkState():
            if self.cat == "Largo":
                if utils.download_largo(self.api, self.vol_id, label, self.work_dir.text()) != 1:
                    print("error")
                    return
            if self.cat == 'Label':
                if utils.download_image(self.api, self.vol_id, label, self.work_dir.text()) != 1:
                    print("error")
                    return 

        self.img_list = utils.list_images(self.work_dir.text())
        self.remaining_nb.setText(str(len(self.img_list)))
        self.timelist = []
        self.prev_timer = time.time()

        self.img_id = 0
        pixmap = QtGui.QPixmap(self.img_list[self.img_id])
        self.image.setPixmap(pixmap)

        self.set_lab1.setText(self.lab1.text())
        self.set_lab2.setText(self.lab2.text())
        if self.lab3.text() == "":
            self.set_lab3.hide()
            self.set_lab3.setDisabled(True)
        else:
            self.set_lab3.setText(self.lab3.text())

        if self.lab4.text() == "":
            self.set_lab4.hide()
            self.set_lab4.setDisabled(True)
        else:
            self.set_lab4.setText(self.lab4.text())

    def nextimage(self, label):
        print(f"Saving Label:{str(label.text())}")
        annotation = os.path.basename(self.img_list[self.img_id])[:-4]
        utils.save_label(self.api, annotation, int(self.origin.text().split("_")[0]), int(label.text().split("_")[0]),
                         self.cat)
        os.remove(self.img_list[self.img_id])

        self.img_id += 1
        if len(self.img_list) > self.img_id:
            self.remaining_nb.setText(str(len(self.img_list) - self.img_id))
            self.done_nb.setText(str(self.img_id))
            self.timelist.append(round((time.time() - self.prev_timer), 2))
            self.prev_timer = time.time()
            self.speed.setText(str(round((sum(self.timelist) / len(self.timelist)) * len(self.img_list) / 60)))

            pixmap = QtGui.QPixmap(self.img_list[self.img_id])
            self.image.setPixmap(pixmap)
        else:
            self.end_class()

    def end_class(self):
        print("Session over !")
        self.origin.setText("")
        self.work_dir.setText("")
        self.lab1.setText("")
        self.lab2.setText("")
        self.lab3.setText("")
        self.lab4.setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())