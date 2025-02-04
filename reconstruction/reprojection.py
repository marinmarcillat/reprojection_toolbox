from UI.meta_reprojection import Ui_Dialog as Ui_Dialog_reproj
from PyQt5.QtWidgets import QDialog, QMessageBox


class rpDialog(QDialog, Ui_Dialog_reproj):
    def __init__(self, qt):
        super().__init__()
        self.setupUi(self)
        self.qt = qt
        self.project_config = qt.project_config

        self.annotations_list = self.project_config['annotations']
        if len(self.annotations_list) != 0:
            item_list = [x['name'] for x in self.annotations_list]
            self.annotation_cb.addItems(item_list)

        self.buttonBox.accepted.clicked.connect(self.launch_reprojection)
        #self.buttonBox.rejected.clicked.connect(self.close)



    def launch_reprojection(self):
        print("Reprojection launching")
        plot_meta = self.plot_cb.isChecked()
        filter_duplicates = self.filter_duplicates_cb.isChecked()
        correct_annotations = self.correct_annotations_cb.isChecked()
        max_time_offset = self.time_offset_sb.value()
        max_reproj_dist = self.max_dist_sb.value()

        rep_name = self.annotation_cb.currentText()
        report = None
        for x in self.annotations_list:
            if x['name'] == rep_name:
                report = x

        if report is not None:
            pass






