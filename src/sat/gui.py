from types import NoneType

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPalette, QBrush

import load_page
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QDialog, QWidget, QMessageBox, \
    QTextEdit, QVBoxLayout, QColorDialog
from PyQt5 import uic
from processor import get_video_config, cv2


class MainWindow(QMainWindow):
    current_video_filepath: pyqtSignal = pyqtSignal(str)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(MainWindow, self).__init__()
        self.setWindowTitle(f"Spectrum Analyzer Tool")
        uic.loadUi("load_page.ui", self)
        self.widget = widget
        self.load_video_btn = self.findChild(QPushButton, "loadVidBtn")
        self.load_video_btn.clicked.connect(self.load_video_btn_callback)
        self.file_location_textbox = self.findChild(QTextEdit, "videoFilepath")

    def load_video_btn_callback(self) -> None:
        fname: QFileDialog = QFileDialog.getOpenFileName(self, "Open File", "", "*.mp4 | *.mkv")
        if fname:
            try:
                filepath: str = fname[0]
                self.go_to_calibration_window(filepath)
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Unable to Load Video:")
                msg.setInformativeText(f"The application was unable to load a video.\n"
                                       f"Error: {e}\n"
                                       f"Ensure that a valid filepath is chosen...")
                msg.setWindowTitle("Error")
                msg.exec_()

    def go_to_calibration_window(self, filepath: str) -> None:
        self.current_video_filepath.emit(filepath)
        self.widget.setCurrentIndex(1)

        # self.widget.show_image(filepath)


class CalibrationWindow(QMainWindow):

    def __init__(self):
        super(CalibrationWindow, self).__init__()
        uic.loadUi("calibration_page.ui", self)
        self.color_picker_btn = self.findChild(QPushButton, "colorPickerBtn")
        self.color_picker_btn.clicked.connect(self.color_picker_btn_callback)
        self.frame_label = self.findChild(QLabel, "frameLabel")
        self.setWindowTitle(f"Calibration Window")
        self.current_video_filepath: str = ''
        self.back_btn = self.findChild(QPushButton, "backBtn")
        self.back_btn.clicked.connect(self.back_btn_callback)
        #QTimer.singleShot(1, self.show_image)  # waits for this to finish until gui displayed
        # self.template_btn = self.findChild(QPushButton, "templateBtn")
        # self.template_btn.clicked.connect(self.template_btn_callback)

    def color_picker_btn_callback(self) -> None:
        dialog = QColorDialog(self)
        dialog.setCurrentColor(Qt.red)
        dialog.exec_()
        value = dialog.currentColor()
        print(value.rgb())

    def get_current_video_filepath(self, signal_filepath) -> None:
        self.current_video_filepath = signal_filepath
        self.show_image()

    @QtCore.pyqtSlot()
    def show_image(self) -> None:
        try:
            print(f"Current Filepath: {self.current_video_filepath}")
            fps, success, frame = get_video_config(filepath=self.current_video_filepath)
            converted_img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
            converted_img = converted_img.scaled(self.frame_label.width(), self.frame_label.height(),
                                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                 QtCore.Qt.TransformationMode.SmoothTransformation)
            self.frame_label.setPixmap(QtGui.QPixmap.fromImage(converted_img))
        except Exception as e:
            print(f"Exception occurred: {e}")

    def back_btn_callback(self) -> None:
        self.widget.setCurrentIndex(0)

    # def template_btn_callback(self) -> None:
    #     fname: QFileDialog = QFileDialog.getOpenFileName(self, "Select Template", "", "*.png | *.jpg | *.jpeg")
    #     selected_fp: str = ""
    #     if fname:
    #         try:
    #             selected_fp = fname[0]
    #         except Exception as e:
    #             msg = QMessageBox()
    #             msg.setIcon(QMessageBox.Critical)
    #             msg.setText("Unable to Select Template File:")
    #             msg.setInformativeText(f"The application was unable to load a video.\n"
    #                                    f"Error: {e}\n"
    #                                    f"Ensure that a valid filepath is chosen...")
    #             msg.setWindowTitle("Error")
    #             msg.exec_()


def start() -> None:
    # initialization of the app
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    load_window = MainWindow(widget=widget)
    cal_window = CalibrationWindow()
    widget.addWidget(load_window)
    widget.addWidget(cal_window)
    widget.setFixedHeight(830)
    widget.setFixedWidth(1277)
    widget.show()
    load_window.current_video_filepath.connect(cal_window.get_current_video_filepath)
    app.exec_()


#: Main entry point
if __name__ == "__main__":
    start()
