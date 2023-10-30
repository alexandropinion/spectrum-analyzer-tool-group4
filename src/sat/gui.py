import configparser
import logging
import os
from types import NoneType
from typing import Optional

import numpy
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPalette, QBrush
from numpy import ndarray

import load_page
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QDialog, QWidget, QMessageBox, \
    QTextEdit, QVBoxLayout, QColorDialog, QLCDNumber, QSlider
from PyQt5 import uic
from processor import get_video_config, cv2, get_frame_count, get_specific_frame

#: Globals
CONFIG_FILENAME: str = 'config.ini'


class MainWindow(QMainWindow):
    current_video_filepath: pyqtSignal = pyqtSignal(str)
    current_csv_filepath: pyqtSignal = pyqtSignal(str)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(MainWindow, self).__init__()
        uic.loadUi("load_page.ui", self)
        self.setWindowTitle(f"Spectrum Analyzer Tool")
        self.widget = widget
        self.load_video_btn = self.findChild(QPushButton, "loadVidBtn")
        self.load_video_btn.clicked.connect(self.load_video_btn_callback)
        self.csv_select_btn = self.findChild(QPushButton, "csvFilepathBtn")
        self.csv_select_btn.clicked.connect(self.csv_select_btn_callback)
        self.csv_textbox = self.findChild(QTextEdit, "csvFilepathLabel")
        self.ini_file_exists: bool = self.load_ini_file_to_app()

    def load_video_btn_callback(self) -> None:
        fname: QFileDialog = QFileDialog.getOpenFileName(self, "Open File", "", "*.mp4 | *.mkv")
        if fname:
            try:
                filepath: str = fname[0]
                conf = configparser.ConfigParser()
                conf.read_file(open(CONFIG_FILENAME, 'r'))
                conf.set("app", "load_video_filepath", filepath)
                with open(CONFIG_FILENAME, "w") as conf_file:
                    conf.write(conf_file)
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

    def load_ini_file_to_app(self) -> bool:
        """
        if ini file doesnt exist, ignore
        :return:
        """
        try:

            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.csv_textbox.setText(config['app']['csv_output_directory'])

        except Exception as e:
            logging.info(e)
            return False

    def csv_select_btn_callback(self) -> None:
        # log_directory: str = filedialog.askdirectory(initialdir=os.getcwd(),
        #                                              title=f"Spectrum Analyzer Tool: Select directory to save CSV results...")
        csv_directory = str(QFileDialog.getExistingDirectory(self, "Select Directory to save CSV data to:"))
        logging.info(f"Location to save CSV file has been selected: {csv_directory}")
        self.current_csv_filepath.emit(csv_directory)
        self.csv_textbox.setText(csv_directory)

        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            conf.set("app", "csv_output_directory", csv_directory)
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)

        except Exception as e:
            logging.info(f"Exception while selecting csv directory: {e}")


class CalibrationWindow(QMainWindow):

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(CalibrationWindow, self).__init__()
        self.widget = widget
        uic.loadUi("calibration_page.ui", self)
        self.color_picker_btn = self.findChild(QPushButton, "colorPickerBtn")
        self.color_picker_btn.clicked.connect(self.color_picker_btn_callback)
        self.frame_label = self.findChild(QLabel, "frameLabel")
        self.setWindowTitle(f"Calibration Window")
        self.current_video_filepath: str = ''
        self.back_btn = self.findChild(QPushButton, "backBtn")
        self.back_btn.clicked.connect(self.back_btn_callback)
        self.red_lcd: QLCDNumber = self.findChild(QLCDNumber, "rColor")
        self.green_lcd: QLCDNumber = self.findChild(QLCDNumber, "gColor")
        self.blue_lcd: QLCDNumber = self.findChild(QLCDNumber, "bColor")
        self.load_rgb_from_ini_config()
        self.frame_slider: QSlider = self.findChild(QSlider, "frameSlider")
        self.frame_slider.valueChanged.connect(self.frame_slider_callback)


    def set_lcd_values(self, r: int, g: int, b: int) -> None:
        self.red_lcd.display(r)
        self.green_lcd.display(g)
        self.blue_lcd.display(b)

    def init_frame_slider(self):
        total_frames: int = get_frame_count(filepath=self.current_video_filepath)
        logging.info(f"Total amount of frames in {self.current_video_filepath}: {total_frames} frames")
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(total_frames)

    def frame_slider_callback(self) -> None:
        frame: ndarray = get_specific_frame(
            filepath=self.current_video_filepath, frame_num=int(self.frame_slider.value()))
        self.load_image_to_frame_label(frame=frame)

    def load_rgb_from_ini_config(self) -> None:
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.set_lcd_values(r=int(config['cal.trace']['red']),
                                    g=int(config['cal.trace']['green']),
                                    b=int(config['cal.trace']['blue']))

        except Exception as e:
            logging.info(f"Error while loading rgb values: {e}")

    def save_rgb_to_init_config(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            conf.set("cal.trace", "red", str(int(self.red_lcd.value())))
            conf.set("cal.trace", "green", str(int(self.green_lcd.value())))
            conf.set("cal.trace", "blue", str(int(self.blue_lcd.value())))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.info(f"Error while saving rgb values to {CONFIG_FILENAME}: {e}")

    def color_picker_btn_callback(self) -> None:
        dialog = QColorDialog(self)
        dialog.setCurrentColor(Qt.red)
        dialog.exec_()
        value = dialog.currentColor()
        logging.info(f"rgb value = {value}\nvalue.rgb() = {value.blue()}")
        self.set_lcd_values(r=value.red(), g=value.green(), b=value.blue())
        try:
            self.save_rgb_to_init_config()
        except Exception as e:
            logging.info(f"Error while selecting color: {e}")

    def get_current_video_filepath(self, signal_filepath) -> None:
        self.current_video_filepath = signal_filepath
        logging.info(f"Current video filepath: {self.current_video_filepath}")
        self.show_image()

    @QtCore.pyqtSlot()
    def show_image(self) -> None:
        try:
            # if get_first_frame:
            #
            # else:
            #     frame_to_show = frame

            fps, success, frame = get_video_config(filepath=self.current_video_filepath)
            self.load_image_to_frame_label(frame=frame)
            self.init_frame_slider()

        except Exception as e:
            logging.info(f"Exception while displaying image: {e}")

    def load_image_to_frame_label(self, frame: ndarray) -> None:
        converted_img = QImage(
            frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
        converted_img = converted_img.scaled(self.frame_label.width(), self.frame_label.height(),
                                             QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.frame_label.setPixmap(QtGui.QPixmap.fromImage(converted_img))

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
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s;%(levelname)s;%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    load_window = MainWindow(widget=widget)
    cal_window = CalibrationWindow(widget=widget)
    widget.addWidget(load_window)
    widget.addWidget(cal_window)
    widget.setFixedHeight(830)
    widget.setFixedWidth(1277)
    widget.show()
    load_window.current_video_filepath.connect(cal_window.get_current_video_filepath)
    # load_window.current_csv_filepath.connect(cal_window.get_current_csv_filepath)
    app.exec_()


#: Main entry point
if __name__ == "__main__":
    start()
