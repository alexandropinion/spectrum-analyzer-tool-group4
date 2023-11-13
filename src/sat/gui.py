import configparser
import logging
import os
from types import NoneType
from typing import Optional, Tuple

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
from processor import get_video_config, cv2, get_frame_count, get_specific_frame, get_reference_frame, \
    crop_template_from_frame, parse_trace_from_frame, load_template_im, show_frame
from distribution import __app_name__, __app_version__

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
        # self.csv_select_btn = self.findChild(QPushButton, "csvFilepathBtn")
        # self.csv_select_btn.clicked.connect(self.csv_select_btn_callback)
        self.csv_textbox = self.findChild(QTextEdit, "csvFilepathLabel")
        self.ini_file_exists: bool = self.load_ini_file_to_app()
        #self.setup_window_backgroud()
        # oImage = QImage("background.jpg")
        # sImage = oImage.scaled(QSize(300, 200))  # resize Image to widgets size
        # palette = QPalette()
        # palette.setBrush(QPalette.Window, QBrush(sImage))
        # self.setPalette(palette)
        #
        # self.label = QLabel('Test', self)  # test, if it's really backgroundimage
        # self.label.setGeometry(50, 50, 200, 50)


    def setup_window_backgroud(self) -> None:
        oImage = QImage("background.jpg")
        sImage = oImage.scaled(QSize(300, 200))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)

        self.label = QLabel('Test', self)  # test, if it's really backgroundimage
        self.label.setGeometry(50, 50, 200, 50)
        self.show()

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

    # def csv_select_btn_callback(self) -> None:
    #     csv_directory = str(QFileDialog.getExistingDirectory(self, "Select Directory to save CSV data to:"))
    #     logging.info(f"Location to save CSV file has been selected: {csv_directory}")
    #     self.current_csv_filepath.emit(csv_directory)
    #     self.csv_textbox.setText(csv_directory)
    #
    #     try:
    #         conf = configparser.ConfigParser()
    #         conf.read_file(open(CONFIG_FILENAME, 'r'))
    #         conf.set("app", "csv_output_directory", csv_directory)
    #         with open(CONFIG_FILENAME, "w") as conf_file:
    #             conf.write(conf_file)
    #
    #     except Exception as e:
    #         logging.info(f"Exception while selecting csv directory: {e}")


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
        self.load_template_btn = self.findChild(QPushButton, "loadTemplateBtn")
        self.load_template_btn.clicked.connect(self.load_template_btn_callback)

    def set_lcd_values(self, r: int, g: int, b: int) -> None:
        self.red_lcd.display(r)
        self.green_lcd.display(g)
        self.blue_lcd.display(b)

    def init_frame_slider(self):
        total_frames: int = get_frame_count(filepath=self.current_video_filepath)
        logging.info(f"Total amount of frames in {self.current_video_filepath}: {total_frames} frames")
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(total_frames)
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            conf.set("cal.trace", "total_frames", str(total_frames))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.info(f"Error while loading rgb values: {e}")

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

    def load_template_btn_callback(self) -> None:
        try:
            self.widget.setCurrentIndex(2)
        except Exception as e:
            pass

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


class TemplateWindow(QMainWindow):

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(TemplateWindow, self).__init__()
        self.window_initialized: bool = False
        self.current_template = []
        self.widget = widget
        uic.loadUi("template_page.ui", self)

        self.next_page_btn = self.findChild(QPushButton, 'nextPageBtn')
        self.next_page_btn.clicked.connect(self.next_page_btn_callback)
        self.template_back_btn = self.findChild(QPushButton, 'templateBackBtn')
        self.template_back_btn.clicked.connect(self.template_back_btn_callback)
        self.load_template_btn = self.findChild(QPushButton, 'loadTemplateBtn')
        self.load_template_btn.clicked.connect(self.load_template_btn_callback)
        self.template_slider: QSlider = self.findChild(QSlider, "templateSlider")
        self.template_slider.valueChanged.connect(self.template_slider_callback)
        self.trace_range_slider: QSlider = self.findChild(QSlider, "traceRangeSlider")
        self.trace_range_slider.sliderReleased.connect(self.trace_range_slider_callback)
        self.current_template_filepath: str = ""  #: TODO: Load from ini file
        self.template_frame_label = self.findChild(QLabel, "processedTemplate")
        self.trace_frame_label = self.findChild(QLabel, "traceTemplate")
        self.current_video_filepath: str = ''

        self.bgra_min_limit = [0, 0, 0, 255]
        self.bgra_max_limit = [0, 0, 0, 255]

        self.init_template_slider()
        self.init_trace_rgb_slider()
        self.init_template_window()
        self.window_initialized = True

    def get_current_video_filepath(self, signal_filepath) -> None:
        self.current_video_filepath = signal_filepath
        logging.info(f"Current video filepath in template window: {self.current_video_filepath}")

    def init_trace_rgb_slider(self) -> None:
        self.trace_range_slider.setMinimum(0)
        self.trace_range_slider.setMaximum(100)
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.trace_range_slider.setValue(int(config['cal.template']['default_trace_slider_val']))
        except Exception as e:
            logging.info(f"Error while initializing trace rgb slider: {e}")
        self.trace_range_slider_callback()

    def init_template_slider(self):
        logging.info(f"Initializing template slider...")
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.template_slider.setMinimum(0)
                self.template_slider.setMaximum(int(config['cal.trace']['total_frames']))
                self.template_slider.setValue(0)
                self.current_template_filepath = str(config['cal.template']['cal_template_filepath'])
                self.current_template.append(load_template_im(template_img_fp=self.current_template_filepath))
        except Exception as e:
            logging.info(f"Error while initializing template slider: {e}")

    @staticmethod
    def process_min_max_trace_range_val(rgb_val: float, trace_percent: float) -> Tuple[int, int]:
        delta_val = int(rgb_val * (trace_percent / 100))
        new_max = rgb_val + delta_val
        new_min = rgb_val - delta_val
        if new_max > 255:
            new_max = 255
        if new_max < 0:
            new_max = 0
        if new_min > 255:
            new_min = 255
        if new_min < 0:
            new_min = 0
        return round(new_min), round(new_max)

    def trace_range_slider_callback(self) -> None:
        try:
            trace_percent = self.trace_range_slider.value()
            if trace_percent > 1.0:
                conf = configparser.ConfigParser()
                conf.read_file(open(CONFIG_FILENAME, 'r'))
                curr_red = int(conf['cal.trace']['red'])
                curr_green = int(conf['cal.trace']['green'])
                curr_blue = int(conf['cal.trace']['blue'])
                new_red_min, new_red_max = self.process_min_max_trace_range_val(rgb_val=curr_red,
                                                                                trace_percent=trace_percent)
                new_green_min, new_green_max = self.process_min_max_trace_range_val(rgb_val=curr_green,
                                                                                    trace_percent=trace_percent)
                new_blue_min, new_blue_max = self.process_min_max_trace_range_val(rgb_val=curr_blue,
                                                                                  trace_percent=trace_percent)
                self.bgra_min_limit = [new_blue_min, new_green_min, new_red_min, 255]
                self.bgra_max_limit = [new_blue_max, new_green_max, new_red_max, 255]

                conf.set("cal.template", "red_max", str(new_red_max))
                conf.set("cal.template", "red_min", str(new_red_min))
                conf.set("cal.template", "green_max", str(new_green_max))
                conf.set("cal.template", "green_min", str(new_green_min))
                conf.set("cal.template", "blue_max", str(new_blue_max))
                conf.set("cal.template", "blue_min", str(new_blue_min))
                conf.set('cal.template', 'default_trace_slider_val', str(self.trace_range_slider.value()))
                with open(CONFIG_FILENAME, "w") as conf_file:
                    conf.write(conf_file)
                if self.window_initialized:
                    self.template_slider_callback()
        except Exception as e:
            logging.info(f"Issue while processing trace range slider: {e}")

    def load_template_frame_to_label(self, frame: ndarray) -> None:
        converted_img = QImage(
            frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
        converted_img = converted_img.scaled(self.template_frame_label.width(), self.template_frame_label.height(),
                                             QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.template_frame_label.setPixmap(QtGui.QPixmap.fromImage(converted_img))

    def load_mask_frame_to_label(self, frame: ndarray) -> None:
        converted_img = QImage(
            frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
        converted_img = converted_img.scaled(self.trace_frame_label.width(), self.trace_frame_label.height(),
                                             QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.trace_frame_label.setPixmap(QtGui.QPixmap.fromImage(converted_img))

    def template_slider_callback(self) -> None:
        try:
            frame: ndarray = get_specific_frame(  # TODO - Get specific "processed" frame
                filepath=self.current_video_filepath, frame_num=int(self.template_slider.value()))

            reference_img, height_template, width_template, size, template_grayscale = (
                get_reference_frame(frame=frame,
                                    template=self.current_template[0]))
            cropped_img = crop_template_from_frame(reference_frame=reference_img,
                                                   template=template_grayscale,
                                                   template_width=width_template,
                                                   template_height=height_template,
                                                   demo=False)

            mask = parse_trace_from_frame(bgra_min_limit=self.bgra_min_limit,
                                          bgra_max_limit=self.bgra_max_limit,
                                          frame=cropped_img)

            qimage_trace = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_Grayscale8)
            height, width, bytesPerComponent = cropped_img.shape
            bgra = numpy.zeros([height, width, 4], dtype=numpy.uint8)
            bgra[:, :, 0:4] = cropped_img
            qimage_template = QImage(bgra.data, width, height, QImage.Format_ARGB32)
            self.trace_frame_label.setPixmap(QtGui.QPixmap.fromImage(qimage_trace))
            self.template_frame_label.setPixmap(QtGui.QPixmap.fromImage(qimage_template))
        except Exception as e:
            logging.info(f"Error while using slider: {e}")

    def next_page_btn_callback(self) -> None:
        self.widget.setCurrentIndex(3)

    def load_template_btn_callback(self) -> None:
        fname: QFileDialog = QFileDialog.getOpenFileName(self, "Select Template Image", "", "*.jpeg | *.png")
        if fname:
            try:
                filepath: str = fname[0]
                self.current_template_filepath = filepath
                conf = configparser.ConfigParser()
                conf.read_file(open(CONFIG_FILENAME, 'r'))
                conf.set("cal.template", "cal_template_filepath", self.current_template_filepath)
                with open(CONFIG_FILENAME, "w") as conf_file:
                    conf.write(conf_file)
                self.current_template[0] = load_template_im(template_img_fp=self.current_template_filepath)
                self.template_slider_callback()
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Unable to Load the Template Image:")
                msg.setInformativeText(f"The application was unable to load a template reference image.\n"
                                       f"Error: {e}\n"
                                       f"Ensure that a valid filepath is chosen...")
                msg.setWindowTitle("Error")
                msg.exec_()

    def init_template_window(self) -> None:
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.current_template_filepath = str(config['cal.template']['cal_template_filepath'])
        except Exception as e:
            logging.info(f"Error while initializing the window page: {e}")

    def template_back_btn_callback(self) -> None:
        self.widget.setCurrentIndex(1)




class SignalWindow(QMainWindow):

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(SignalWindow, self).__init__()
        self.window_initialized: bool = False
        self.current_template = []
        self.widget = widget
        uic.loadUi("signal_page.ui", self)
        self.current_video_filepath: str = ""

    def get_current_video_filepath(self, signal_filepath) -> None:
        self.current_video_filepath = signal_filepath
        logging.info(f"Current video filepath in template window: {self.current_video_filepath}")


    def start_processing_btn_callback(self) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"Are you sure you want to start processing?")  # TODO - Add estimated time?
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        val = msg.exec_()
        if val == QMessageBox.Ok:
            self.start_processing()

    def start_processing(self) -> None:
        pass



def start() -> None:
    # initialization of the app
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s;%(levelname)s;%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    app = QApplication(sys.argv)
    app.setApplicationName(f"{__app_name__} {__app_version__}")
    widget = QtWidgets.QStackedWidget()
    load_window = MainWindow(widget=widget)
    cal_window = CalibrationWindow(widget=widget)
    template_window = TemplateWindow(widget=widget)
    signal_window = SignalWindow(widget=widget)
    widget.addWidget(load_window)
    widget.addWidget(cal_window)
    widget.addWidget(template_window)
    widget.addWidget(signal_window)
    widget.setFixedHeight(830)
    widget.setFixedWidth(1277)
    widget.show()
    load_window.current_video_filepath.connect(cal_window.get_current_video_filepath)
    load_window.current_video_filepath.connect(template_window.get_current_video_filepath)
    load_window.current_video_filepath.connect(signal_window.get_current_video_filepath)
    # load_window.current_csv_filepath.connect(cal_window.get_current_csv_filepath)
    app.exec_()


#: Main entry point
if __name__ == "__main__":
    start()
