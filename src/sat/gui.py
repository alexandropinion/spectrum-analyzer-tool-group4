#!/usr/bin/env python3

#: Imports
import logging
import configparser
import os
import queue
import threading
import sys
import numpy
from typing import Tuple
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
from numpy import ndarray
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QMessageBox, \
    QTextEdit, QColorDialog, QLCDNumber, QSlider, QToolButton, QCheckBox, QProgressBar, QPlainTextEdit, \
    QDial
from PyQt5 import uic
from processor import get_video_config, get_frame_count_and_fps, get_specific_frame, get_reference_frame, \
    crop_template_from_frame, parse_trace_from_frame, load_template_im, read_signal_levels_from_frame, \
    get_preprocessed_image_for_text_detection, Processor
from distribution import __app_name__, __app_version__
from saattype import ProcessorParams, get_datetime_heading

#: Globals
CONFIG_FILENAME: str = 'config.ini'
DEFAULT_APP_WIDTH: int = 1277
DEFAULT_APP_HEIGHT: int = 830
DEFAULT_MAIN_WINDOW_WIDTH: int = 800
DEFAULT_MAIN_WINDOW_HEIGHT: int = 600


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


class MainWindow(QMainWindow):
    current_video_filepath_signal: pyqtSignal = pyqtSignal(str)
    current_csv_filepath_signal: pyqtSignal = pyqtSignal(str)
    move_to_cal_window_signal: pyqtSignal = pyqtSignal(bool)
    start_processing_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(MainWindow, self).__init__()
        uic.loadUi(resource_path("load_page.ui"), self)
        self.setWindowTitle(f"Spectrum Analyzer Tool")
        self.current_loaded_video_filepath: str = ""
        self.widget = widget
        self.load_video_btn = self.findChild(QPushButton, "loadVidBtn")
        self.load_video_btn.setStyleSheet("background-color : #D8F5FF")
        self.load_video_btn.clicked.connect(self.load_video_btn_callback)
        self.start_processor_preset_btn = self.findChild(QToolButton, "startProcessorPresets")
        self.start_processor_preset_btn.setStyleSheet("background-color : #D0D0D0")
        self.start_processor_preset_btn.clicked.connect(self.start_processor_preset_btn_callback)
        self.calibrate_processor_preset_btn = self.findChild(QToolButton, "calibrateProcessorPresets")
        self.calibrate_processor_preset_btn.setStyleSheet("background-color : #D0D0D0")
        self.calibrate_processor_preset_btn.clicked.connect(self.calibrate_processor_preset_btn_callback)
        self.calibrate_processor_preset_btn.setDisabled(True)
        self.start_processor_preset_btn.setDisabled(True)
        # self.csv_textbox = self.findChild(QTextEdit, "csvFilepathLabel")
        # self.ini_file_exists: bool = self.load_ini_file_to_app()
        self.setup_window_backgroud()

    def start_processor_preset_btn_callback(self) -> None:
        start: bool = confirm(msg=f"Are you sure you want to begin processing with all of the preset values "
                                  f"configured from the previous run?")
        if start:
            logging.info(f"Load window has requested to start processing with presets - moving to progress window...")
            self.start_processing_signal.emit(True)
            self.widget.setFixedWidth(DEFAULT_APP_WIDTH)
            self.widget.setFixedHeight(DEFAULT_APP_HEIGHT)
            self.widget.setCurrentIndex(4)

    def calibrate_processor_preset_btn_callback(self) -> None:
        self.go_to_calibration_window(filepath=self.current_loaded_video_filepath)

    def setup_window_backgroud(self) -> None:
        self.setFixedWidth(DEFAULT_MAIN_WINDOW_WIDTH)
        self.setFixedHeight(DEFAULT_MAIN_WINDOW_HEIGHT)
        stylesheet = '''
            #MainWindow {
                background-image: url(background.jpeg);
                background-repeat: no-repeat;
                background-position: center;
            }
        '''
        self.setStyleSheet(stylesheet)
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
                self.current_loaded_video_filepath = filepath
                self.calibrate_processor_preset_btn.setDisabled(False)
                self.start_processor_preset_btn.setDisabled(False)
                self.start_processor_preset_btn.setStyleSheet("background-color : #98F7C3")
                self.calibrate_processor_preset_btn.setStyleSheet("background-color : #82FFF9")
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
        self.current_video_filepath_signal.emit(filepath)
        self.move_to_cal_window_signal.emit(True)
        self.widget.setCurrentIndex(1)

    # def load_ini_file_to_app(self) -> bool:
    #     try:
    #         with open(f"{CONFIG_FILENAME}") as config_file:
    #             config = configparser.ConfigParser()
    #             config.read_file(config_file)
    #             self.csv_textbox.setText(config['app']['csv_output_directory'])
    #     except Exception as e:
    #         logging.info(e)
    #         return False


class CalibrationWindow(QMainWindow):
    go_to_template_window_signal: pyqtSignal = pyqtSignal(bool)
    loaded_new_video_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(CalibrationWindow, self).__init__()
        self.widget = widget
        uic.loadUi(resource_path("calibration_page.ui"), self)
        self.color_picker_btn = self.findChild(QPushButton, "colorPickerBtn")
        self.color_picker_btn.clicked.connect(self.color_picker_btn_callback)
        self.frame_label = self.findChild(QLabel, "frameLabel")
        self.setWindowTitle(f"Calibration Window")
        self.current_video_filepath: str = ''
        self.back_btn = self.findChild(QPushButton, "backBtn")
        self.back_btn.clicked.connect(self.back_btn_callback)

        self.back_btn.setStyleSheet("background-color : #FF9C6D")
        self.red_lcd: QLCDNumber = self.findChild(QLCDNumber, "rColor")
        self.green_lcd: QLCDNumber = self.findChild(QLCDNumber, "gColor")
        self.blue_lcd: QLCDNumber = self.findChild(QLCDNumber, "bColor")
        self.load_rgb_from_ini_config()
        self.frame_slider: QSlider = self.findChild(QSlider, "frameSlider")
        self.frame_slider.valueChanged.connect(self.frame_slider_callback)
        self.load_template_btn = self.findChild(QPushButton, "loadTemplateBtn")
        self.load_template_btn.clicked.connect(self.load_template_btn_callback)
        self.load_template_btn.setStyleSheet("background-color : #62FFAD")

    def set_lcd_values(self, r: int, g: int, b: int) -> None:
        self.red_lcd.display(r)
        self.green_lcd.display(g)
        self.blue_lcd.display(b)

    def init_frame_slider(self):
        total_frames, fps = get_frame_count_and_fps(filepath=self.current_video_filepath)
        logging.info(f"Total amount of frames in {self.current_video_filepath}: {total_frames} frames\n"
                     f"Total FPS: {fps}")
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(total_frames)
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            conf.set("cal.trace", "total_frames", str(total_frames))
            conf.set("cal.signal", "maximum_frames_to_process_per_second", str(fps))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
                # conf_file.close()
            self.loaded_new_video_signal.emit(True)
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
            self.go_to_template_window_signal.emit(True)
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
    def moved_to_cal_window(self) -> None:
        try:
            self.widget.setFixedWidth(DEFAULT_APP_WIDTH)
            self.widget.setFixedHeight(DEFAULT_APP_HEIGHT)
        except Exception as e:
            logging.info(f"Exception while displaying image: {e}")

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
        self.widget.setFixedWidth(DEFAULT_MAIN_WINDOW_WIDTH)
        self.widget.setFixedHeight(DEFAULT_MAIN_WINDOW_HEIGHT)
        self.widget.setCurrentIndex(0)


class TemplateWindow(QMainWindow):
    go_to_signal_window_signal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(TemplateWindow, self).__init__()
        self.window_initialized: bool = False
        self.current_template = []
        self.widget = widget
        uic.loadUi(resource_path("template_page.ui"), self)

        self.next_page_btn = self.findChild(QPushButton, 'nextPageBtn')
        self.next_page_btn.clicked.connect(self.next_page_btn_callback)
        self.next_page_btn.setStyleSheet("background-color : #62FFAD")
        self.template_back_btn = self.findChild(QPushButton, 'templateBackBtn')
        self.template_back_btn.clicked.connect(self.template_back_btn_callback)
        self.template_back_btn.setStyleSheet("background-color : #FF9C6D")
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

    @QtCore.pyqtSlot()
    def load_processed_images_to_window(self) -> None:
        try:
            self.template_slider_callback()
        except Exception as e:
            logging.info(f"Exception while displaying image: {e}")

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
            cropped_img, inverse_cropped_img = \
                crop_template_from_frame(reference_frame=reference_img,
                                         template=template_grayscale,
                                         template_width=width_template,
                                         template_height=height_template)

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
        self.go_to_signal_window_signal.emit(True)
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
    start_processor_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(SignalWindow, self).__init__()
        self.window_initialized: bool = False
        self.current_template = []
        self.widget = widget
        uic.loadUi(resource_path("signal_page.ui"), self)
        self.current_video_filepath: str = ""
        self.current_template_filepath: str = ""
        self.current_text_img_threshold: int = 0
        self.total_frames_in_video: int = 0
        self.signal_frame_label = self.findChild(QLabel, "signalImage")
        self.start_processor_btn = self.findChild(QPushButton, 'startProcessingBtn')
        self.start_processor_btn.clicked.connect(self.start_processing_btn_callback)
        self.go_back_btn = self.findChild(QPushButton, 'goBackBtn')
        self.go_back_btn.clicked.connect(self.go_back_btn_callback)
        self.scan_text_slider: QSlider = self.findChild(QSlider, "selectTextScanSlider")
        self.scan_text_slider.sliderReleased.connect(self.scan_text_slider_callback)
        self.record_freq_dbm_checkbox = self.findChild(QCheckBox, "recordFreqDbm")
        self.record_max_ampl_checkbox = self.findChild(QCheckBox, "recordMaxAmpl")
        self.record_relative_ampl_checkbox = self.findChild(QCheckBox, "recordRelativeSignal")
        self.record_freq_dbm_checkbox.clicked.connect(self.record_freq_dbm_checkbox_callback)
        self.record_max_ampl_checkbox.clicked.connect(self.record_max_ampl_checkbox_callback)
        self.record_relative_ampl_checkbox.clicked.connect(self.record_relative_ampl_checkbox_callback)
        self.select_csv_filepath = self.findChild(QPushButton, "csvFilepathBtn")
        self.select_csv_filepath.clicked.connect(self.select_csv_filepath_callback)
        self.select_csv_filepath_text = self.findChild(QTextEdit, "csvFilepathLabel")
        self.select_fps_dial: QSlider = self.findChild(QDial, "frameDial")
        self.select_fps_dial.sliderReleased.connect(self.select_fps_dial_callback)
        self.select_fps_lcd: QLCDNumber = self.findChild(QLCDNumber, "frameLcd")

        # Signal IDs
        self.center_freq_id = self.findChild(QTextEdit, "centerFreqID")
        self.span_freq_id = self.findChild(QTextEdit, "spanFreqID")
        self.ref_level_id = self.findChild(QTextEdit, "ref_level_ID")
        self.division_slider: QSlider = self.findChild(QSlider, "divisionSlider")
        self.division_slider.sliderReleased.connect(self.division_slider_callback)
        self.division_lcd: QLCDNumber = self.findChild(QLCDNumber, "divisionLcd")
        self.division_per_db_lcd: QLCDNumber = self.findChild(QLCDNumber, "dbPerDivLED")
        self.threshold_lcd = self.findChild(QLCDNumber, "thresholdLcd")
        self.scan_threshold_checkbox = self.findChild(QCheckBox, "scanThreshold")
        self.scan_threshold_checkbox.clicked.connect(self.scan_threshold_checkbox_callback)
        self.scan_threshold_slider: QSlider = self.findChild(QSlider, "thresholdSlider")
        self.scan_threshold_slider.sliderReleased.connect(self.scan_threshold_slider_callback)
        self.division_per_db_slider: QSlider = self.findChild(QSlider, "divisionPerDbSlider")
        self.division_per_db_slider.sliderReleased.connect(self.division_per_db_slider_callback)
        self.select_text_img_threshold_slider: QSlider = self.findChild(QSlider, "selectThresholdSlider")
        self.select_text_img_threshold_slider.sliderReleased.connect(self.select_text_img_threshold_slider_callback)
        self.text_img_threshold_lcd: QLCDNumber = self.findChild(QLCDNumber, "imgTextThreshold")

        # LCDs for signal characteristics
        self.center_freq_lcd: QLabel = self.findChild(QLabel, "centerFreq")
        self.span_freq_lcd: QLabel = self.findChild(QLabel, "spanFreq")
        self.ref_lvl_lcd: QLabel = self.findChild(QLabel, "refLevel")
        self.current_center_freq: int = 0
        self.current_span_freq: int = 0
        self.current_ref_lvl: int = 0

        self.init_sliders()
        self.initialize_signal_window()

    def select_fps_dial_callback(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val = int(self.select_fps_dial.value())
            self.select_fps_lcd.display(val)
            conf.set("cal.signal", "frames_to_process_per_second", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
            self.current_text_img_threshold = val
        except Exception as e:
            logging.critical(f"Issue while moving fps dial: {e}")

    def select_text_img_threshold_slider_callback(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val = int(self.select_text_img_threshold_slider.value())
            self.text_img_threshold_lcd.display(val)
            conf.set("cal.signal", "text_image_threshold", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
            self.current_text_img_threshold = val
            self.scan_text_slider_callback()
        except Exception as e:
            logging.critical(f"Issue while selecting text image threshold slider: {e}")

    def division_per_db_slider_callback(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val = self.division_per_db_slider.value()
            self.division_per_db_lcd.display(int(val))
            conf.set("cal.signal", "db_per_division", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue while using division per slider callback: {e}")

    def division_slider_callback(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val = self.division_slider.value()
            self.division_lcd.display(val)
            conf.set("cal.signal", "total_db_divisions", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning division slider: {e}")

    def scan_threshold_slider_callback(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val = self.scan_threshold_slider.value()
            self.threshold_lcd.display(val)
            conf.set("cal.signal", "threshold_percent", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning threshold slider: {e}")

    def init_sliders(self) -> None:
        self.scan_threshold_slider.setMinimum(0)
        self.scan_threshold_slider.setMaximum(100)
        self.division_slider.setMinimum(5)
        self.division_slider.setMaximum(20)
        self.division_per_db_slider.setMaximum(50)
        self.division_per_db_slider.setMinimum(1)
        self.select_text_img_threshold_slider.setMinimum(0)
        self.select_text_img_threshold_slider.setMaximum(255)
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.division_per_db_slider.setValue(int(config['cal.signal']['db_per_division']))
                self.division_per_db_lcd.display(int(config['cal.signal']['db_per_division']))
                self.division_slider.setValue(int(config['cal.signal']['total_db_divisions']))
                self.division_lcd.display(int(config['cal.signal']['total_db_divisions']))
                self.scan_threshold_slider.setValue(int(config['cal.signal']['threshold_percent']))
                self.threshold_lcd.display(int(config['cal.signal']['threshold_percent']))
                total_frames = int(config['cal.trace']['total_frames'])
                self.total_frames_in_video = total_frames
                self.scan_text_slider.setMinimum(0)
                self.scan_text_slider.setMaximum(total_frames)
                fps_max = int(config['cal.signal']['maximum_frames_to_process_per_second'])
                current_fps = int(config['cal.signal']['frames_to_process_per_second'])
                self.select_fps_dial.setMinimum(1)
                self.select_fps_dial.setMaximum(fps_max)
                self.select_fps_dial.setValue(current_fps)
                self.select_fps_lcd.display(current_fps)
        except Exception as e:
            logging.info(f"Error while initializing trace rgb slider: {e}")

    @QtCore.pyqtSlot()
    def update_ui_from_new_video_loaded(self) -> None:
        try:

            config = configparser.ConfigParser()
            config.read_file(open(CONFIG_FILENAME, 'r'))
            curr_max_fps = int(config['cal.signal']['maximum_frames_to_process_per_second'])
            curr_fps_selected = int(config['cal.signal']['frames_to_process_per_second'])
            if curr_fps_selected > curr_max_fps:
                config.set("cal.signal", "frames_to_process_per_second", str(curr_max_fps))
                self.select_fps_dial.setValue(1)
                self.select_fps_dial.setMaximum(curr_max_fps)
                self.select_fps_dial.setValue(curr_max_fps)
                self.select_fps_lcd.display(curr_max_fps)
            config.set("cal.signal", "maximum_frames_to_process_per_second", str(curr_max_fps))
            with open(f"{CONFIG_FILENAME}", "w") as config_file:
                config.write(config_file)
        except Exception as e:
            logging.critical(f"Issue while updating the ui due to a new video being loaded: {e}")

    @QtCore.pyqtSlot()
    def signal_window_is_being_shown(self) -> None:
        try:
            self.init_sliders()
            self.initialize_signal_window()
            self.select_text_img_threshold_slider_callback()
        except Exception as e:
            logging.info(f"Exception while displaying image: {e}")

    def scan_threshold_checkbox_callback(self) -> None:
        is_checked = self.scan_threshold_checkbox.isChecked()
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val: int = 0
            if is_checked:
                val = 1
            conf.set("cal.signal", "scan_relative_amplitude_threshold", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning threshold checkbox: {e}")

    def select_csv_filepath_callback(self) -> None:
        try:
            csv_directory = str(QFileDialog.getExistingDirectory(self, "Select Directory to save CSV data to:"))
            logging.info(f"Location to save CSV file has been selected: {csv_directory}")
            self.select_csv_filepath_text.setPlainText(csv_directory)
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            conf.set("cal.signal", "csv_output_directory", csv_directory)
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.info(f"Exception while selecting csv directory: {e}")

    def record_freq_dbm_checkbox_callback(self) -> None:
        is_checked = self.record_freq_dbm_checkbox.isChecked()
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val: int = 0
            if is_checked:
                val = 1
            else:
                if not self.record_max_ampl_checkbox.isChecked() and not self.record_relative_ampl_checkbox.isChecked():
                    val = 1
                    self.record_freq_dbm_checkbox.setChecked(True)  # Must have at least one checkbox selected
            conf.set("cal.signal", "record_entire_signal_with_scaling_factors", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning threshold checkbox: {e}")

    def record_max_ampl_checkbox_callback(self) -> None:
        is_checked = self.record_max_ampl_checkbox.isChecked()
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val: int = 0
            if is_checked:
                val = 1
            else:
                if not self.record_freq_dbm_checkbox.isChecked() and not self.record_relative_ampl_checkbox.isChecked():
                    val = 1
                    self.record_max_ampl_checkbox.setChecked(True)  # Must have at least one checkbox selected
            conf.set("cal.signal", "record_only_max_signal_with_scaling_factors", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning threshold checkbox: {e}")

    def record_relative_ampl_checkbox_callback(self) -> None:
        is_checked = self.record_relative_ampl_checkbox.isChecked()
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            val: int = 0
            if is_checked:
                val = 1
            else:
                if not self.record_freq_dbm_checkbox.isChecked() and not self.record_max_ampl_checkbox.isChecked():
                    val = 1
                    self.record_relative_ampl_checkbox.setChecked(True)  # Must have at least one checkbox selected
            conf.set("cal.signal", "record_entire_signal_with_no_scaling_factors", str(val))
            with open(CONFIG_FILENAME, "w") as conf_file:
                conf.write(conf_file)
        except Exception as e:
            logging.critical(f"Issue scanning threshold checkbox: {e}")

    def scan_text_slider_callback(self) -> None:
        try:
            center_freq_id = self.center_freq_id.toPlainText()
            span_freq_id = self.span_freq_id.toPlainText()
            ref_level_id = self.ref_level_id.toPlainText()
            frame = get_specific_frame(filepath=self.current_video_filepath,
                                       frame_num=int(self.scan_text_slider.value()))
            success, ref_lvl, center_freq, span_freq = read_signal_levels_from_frame(
                frame=frame, center_freq_tag=center_freq_id,
                reference_level_tag=ref_level_id, span_tag=span_freq_id,
                text_img_threshold=self.current_text_img_threshold)

            frame_to_show = get_preprocessed_image_for_text_detection(
                frame=frame, threshold=self.current_text_img_threshold)

            converted_img = QImage(
                frame_to_show, frame_to_show.shape[1],
                frame_to_show.shape[0], QImage.Format.Format_Grayscale8)
            converted_img = converted_img.scaled(self.signal_frame_label.width(), self.signal_frame_label.height(),
                                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                 QtCore.Qt.TransformationMode.SmoothTransformation)
            self.signal_frame_label.setPixmap(QtGui.QPixmap.fromImage(converted_img))
            if success:
                logging.info(f"success reading levels: {success}\n"
                             f"ref lvl = {ref_lvl}\n"
                             f"center freq = {center_freq}\n"
                             f"span freq = {span_freq}")
                self.set_label_to_digits(value=center_freq / 1e9, label=self.center_freq_lcd)
                self.set_label_to_digits(value=span_freq / 1e9, label=self.span_freq_lcd)
                self.set_label_to_digits(value=ref_lvl / 1e9, label=self.ref_lvl_lcd)
                self.current_center_freq = center_freq
                self.current_span_freq = span_freq
                self.current_ref_lvl = ref_lvl

        except Exception as e:
            logging.critical(f"Issue while scanning text slider: {e}")

    @staticmethod
    def set_label_to_digits(value: float, label: QLabel) -> None:
        label.setText('{:.02f}'.format(value))

    def go_back_btn_callback(self) -> None:
        self.widget.setCurrentIndex(2)

    def initialize_signal_window(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            text_img_threshold = int(conf['cal.signal']['text_image_threshold'])
            db_per_division = int(conf['cal.signal']['db_per_division'])
            center_freq_text = str(conf['cal.signal']['center_freq_text'])
            reference_level_text = str(conf['cal.signal']['reference_level_text'])
            span_text = str(conf['cal.signal']['span_text'])
            total_db_divisions = int(conf['cal.signal']['total_db_divisions'])
            csv_output_directory = str(conf['cal.signal']['csv_output_directory'])
            record_entire_signal_with_scaling_factors = (
                int(conf['cal.signal']['record_entire_signal_with_scaling_factors']))
            record_only_max_signal_with_scaling_factors = (
                int(conf['cal.signal']['record_only_max_signal_with_scaling_factors']))
            record_entire_signal_with_no_scaling_factors = (
                int(conf['cal.signal']['record_entire_signal_with_no_scaling_factors']))
            scan_relative_amplitude_threshold = (
                int(conf['cal.signal']['scan_relative_amplitude_threshold']))
            if record_entire_signal_with_no_scaling_factors == 1:
                self.record_relative_ampl_checkbox.setChecked(True)
            else:
                self.record_relative_ampl_checkbox.setChecked(False)
            if record_only_max_signal_with_scaling_factors == 1:
                self.record_max_ampl_checkbox.setChecked(True)
            else:
                self.record_max_ampl_checkbox.setChecked(False)
            if record_entire_signal_with_scaling_factors == 1:
                self.record_freq_dbm_checkbox.setChecked(True)
            else:
                self.record_freq_dbm_checkbox.setChecked(False)
            if scan_relative_amplitude_threshold == 1:
                self.scan_threshold_checkbox.setChecked(True)
            else:
                self.scan_threshold_checkbox.setChecked(False)

            self.current_text_img_threshold = text_img_threshold  # TODO: update threshold lcd and slider
            self.text_img_threshold_lcd.display(text_img_threshold)
            self.select_text_img_threshold_slider.setValue(text_img_threshold)
            self.select_csv_filepath_text.setText(csv_output_directory)
            self.center_freq_id.setText(center_freq_text)
            self.span_freq_id.setText(span_text)
            self.ref_level_id.setText(reference_level_text)
            self.division_slider.setValue(total_db_divisions)
            self.division_per_db_slider.setValue(db_per_division)
            # self.division_per_db_lcd.dislay(db_per_division)
            self.select_csv_filepath.setStyleSheet("background-color : #B2FDFF")
            self.start_processor_btn.setStyleSheet("background-color : #62FFAD")
            self.go_back_btn.setStyleSheet("background-color : #FF9C6D")
            self.current_template_filepath = str(conf['cal.template']['cal_template_filepath'])
            self.current_template.append(load_template_im(template_img_fp=self.current_template_filepath))

        except Exception as e:
            logging.critical(f"Issue while initializing the signal window: {e}")

    def get_current_video_filepath(self, signal_filepath) -> None:
        self.current_video_filepath = signal_filepath
        logging.info(f"Current video filepath in template window: {self.current_video_filepath}")

    def start_processing_btn_callback(self) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"Are you sure you want to start processing?")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        val = msg.exec_()
        if val == QMessageBox.Ok:
            self.start_processor_signal.emit(True)
            self.widget.setCurrentIndex(4)


class ProgressWindow(QMainWindow):
    finished_processing_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(ProgressWindow, self).__init__()
        self.window_initialized: bool = False
        self.widget = widget
        uic.loadUi(resource_path("progress_page.ui"), self)
        self.progress_bar = self.findChild(QProgressBar, "progressBar")
        self.cancel_btn = self.findChild(QPushButton, "cancelBtn")
        self.cancel_btn.setStyleSheet("background-color : #FFC2B9")
        self.plain_text_box = self.findChild(QPlainTextEdit, "progressText")
        self.cancel_btn.clicked.connect(self.cancel_btn_callback)
        self.data_q: queue.Queue = None
        self.total_frames: int = None
        self.current_params: ProcessorParams = None
        self.current_processor_thread: ProcessorThread = None
        stylesheet = '''
                    #MainWindow {
                        background-image: url(file:///''' + resource_path('loading_background.jpg') + ''');
                        background-repeat: no-repeat;
                        background-position: center;
                    }
                '''
        self.setStyleSheet(stylesheet)

    @QtCore.pyqtSlot()
    def start_processing(self) -> None:
        self.widget.setFixedWidth(DEFAULT_APP_WIDTH)
        self.widget.setFixedHeight(DEFAULT_APP_HEIGHT)
        try:
            success, self.current_params = get_all_processor_params_from_ini(ini_fp=CONFIG_FILENAME)
            self.update_total_frames()
            self.plain_text_box.appendPlainText(
                f"{get_datetime_heading()}: "
                f"Processing the following video file: {self.current_params.video_fp}\n"
                f"{get_datetime_heading()}: Logs are being stored in: {self.current_params.log_directory}")

            if success:
                self.data_q: queue.Queue = ProcessorQueueCallback(callback=self.data_queue_callback)
                self.current_processor_thread = ProcessorThread(params=self.current_params, data_queue=self.data_q)
                self.current_processor_thread.daemon = False
                self.current_processor_thread.start()
            else:
                prompt(
                    msg=f"Issue loading ini file.\n"
                        f"Check values and file location.\n"
                        f"config.ini file should exist within the same directory as the executable.", error=True)
                self.widget.setCurrentIndex(0)
        except Exception as e:
            logging.critical(f"Issue starting processing thread: {e}")

    @QtCore.pyqtSlot()
    def finished(self) -> None:
        try:
            self.progress_bar.setValue(0)
            logging.info(f"Finished signal has been emitted")
            prompt(msg=f"Processing has finished! CSV log results will be stored in"
                       f"the following directory: {self.current_params.log_directory}",
                   error=False)
        except Exception as e:
            logging.critical(f"Error while finishing processor in progress window: {e}")

    def update_total_frames(self) -> None:
        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            self.total_frames = int(conf['cal.trace']['total_frames'])
        except Exception as e:
            logging.critical(f"Issue while reading ini file: {e}")

    def data_queue_callback(self, data) -> None:
        try:
            logging.debug(f"Data received from data queue: {data}")
            curr_frame = data.current_frame
            progress_bar_val = int((curr_frame / self.total_frames) * 100)
            if progress_bar_val > 100:
                progress_bar_val = 100
            self.progress_bar.setValue(progress_bar_val)
            if data.finished:
                logging.critical(f"Finished processing...")
                self.plain_text_box.appendPlainText(
                    f"{get_datetime_heading()}: Video file processing ending...")
                self.finished_processing_signal.emit(True)
        except Exception as e:
            logging.critical(f"Error while calling data queue callback: {e}")

    def cancel_btn_callback(self) -> None:
        if confirm(msg=f"Are you sure you want to return to Main Menu?\n\n"
                       f"If a video is being processed, it will be terminated."):
            self.progress_bar.setValue(0)
            try:
                self.current_processor_thread.stop()
            except Exception as e:
                logging.critical(f"No thread running...{e}")
            self.widget.setFixedWidth(DEFAULT_MAIN_WINDOW_WIDTH)
            self.widget.setFixedHeight(DEFAULT_MAIN_WINDOW_HEIGHT)
            self.widget.setCurrentIndex(0)


def get_all_processor_params_from_ini(ini_fp: str) -> Tuple[bool, ProcessorParams] | Tuple[bool, None]:
    logging.info(f"Requested config filepath: {ini_fp}")
    try:
        conf = configparser.ConfigParser()
        params: ProcessorParams = None
        with open(ini_fp, 'r') as config_file:
            conf.read_file(config_file)
            # conf.read_file(open(ini_fp, 'r'))
            params = ProcessorParams(video_fp=conf['app']['load_video_filepath'],
                                     template_fp=conf['cal.template']['cal_template_filepath'],
                                     log_directory=conf['cal.signal']['csv_output_directory'],
                                     dbm_magnitude_threshold=float(conf['cal.signal']['threshold_percent']) / 100.0,
                                     bgra_min_filter=[int(conf['cal.template']['blue_min']),
                                                      int(conf['cal.template']['green_min']),
                                                      int(conf['cal.template']['red_min']),
                                                      255],
                                     bgra_max_filter=[int(conf['cal.template']['blue_max']),
                                                      int(conf['cal.template']['green_max']),
                                                      int(conf['cal.template']['red_max']),
                                                      255],
                                     total_num_db_divisions=int(conf['cal.signal']['total_db_divisions']),
                                     frames_to_read_per_sec=int(conf['cal.signal']['frames_to_process_per_second']),
                                     db_per_division=int(conf['cal.signal']['db_per_division']),
                                     text_img_threshold=int(conf['cal.signal']['text_image_threshold']),
                                     center_freq_id=conf['cal.signal']['center_freq_text'],
                                     ref_lvl_id=conf['cal.signal']['reference_level_text'],
                                     span_lvl_id=conf['cal.signal']['span_text'],
                                     scan_for_relative_threshold=bool(
                                         int(conf['cal.signal']['scan_relative_amplitude_threshold'])),
                                     record_relative_signal=bool(
                                         int(conf['cal.signal']['record_entire_signal_with_no_scaling_factors'])),
                                     record_scaled_signal=bool(
                                         int(conf['cal.signal']['record_entire_signal_with_scaling_factors'])),
                                     record_max_signal_scaled=bool(
                                         int(conf['cal.signal']['record_only_max_signal_with_scaling_factors'])))

        return True, params
    except Exception as e:
        logging.critical(f"Issue while grabbing all processor params from {ini_fp}: {e}")
        return False, None


def confirm(msg: str) -> bool:
    try:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(msg)
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        val = msg_box.exec_()
        if val == QMessageBox.Ok:
            return True
        else:
            return False
    except Exception as e:
        logging.critical(f"Issue while using confirmation box: {e}")
        return False


def prompt(msg: str, error: bool) -> None:
    try:
        msg_box = QMessageBox()
        if error:
            msg_type = QMessageBox.Critical
        else:
            msg_type = QMessageBox.Information
        msg_box.setIcon(msg_type)
        msg_box.setText(msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        val = msg_box.exec_()
    except Exception as e:
        logging.critical(f"Issue while using confirmation box: {e}")
        return False


#: Classes
class ProcessorQueueCallback(queue.Queue):
    def __init__(self, callback=None, maxsize=0):
        super().__init__(maxsize=maxsize)
        self.callback = callback

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        if self.callback:
            logging.debug(f"Writing data to the queue")
            if self.qsize() > 10000:
                while True:
                    try:
                        logging.debug(f"Attempting to clear the queue")
                        self.get_nowait()
                    except (Exception, queue.Empty) as e:
                        logging.critical(f"Error reported while clearing queue: {e}")
                        break
            self.callback(item)


class ProcessorThread(threading.Thread):
    def __init__(self, params: ProcessorParams, data_queue: queue.Queue):
        super(ProcessorThread, self).__init__()
        self.stop_thread = threading.Event()
        self.data_queue = data_queue
        self.params = params

    def run(self) -> None:
        processor = Processor(video_fp=self.params.video_fp,
                              template_fp=self.params.template_fp,
                              bgra_max_filter=self.params.bgra_max_filter,
                              bgra_min_filter=self.params.bgra_min_filter,
                              db_per_division=self.params.db_per_division,
                              dbm_magnitude_threshold=self.params.dbm_magnitude_threshold,
                              log_directory=self.params.log_directory,
                              record_max_signal_scaled=self.params.record_max_signal_scaled,
                              record_scaled_signal=self.params.record_scaled_signal,
                              record_relative_signal=self.params.record_relative_signal,
                              frames_to_process_per_s=self.params.frames_to_read_per_sec,
                              ref_lvl_id=self.params.ref_lvl_id,
                              scan_for_relative_threshold=self.params.scan_for_relative_threshold,
                              span_lvl_id=self.params.span_lvl_id,
                              center_freq_id=self.params.center_freq_id,
                              text_img_threshold=self.params.text_img_threshold,
                              stop_event=self.stop_thread,
                              data_q=self.data_queue,
                              total_db_divisions=self.params.total_num_db_divisions)
        processor.run()

    def stop(self) -> None:
        logging.critical(f"Requesting to stop the current processing thread...")
        self.stop_thread.set()


def start() -> None:
    # initialization of the app
    app = QApplication(sys.argv)
    app.setApplicationName(f"{__app_name__} {__app_version__}")
    widget = QtWidgets.QStackedWidget()
    load_window = MainWindow(widget=widget)
    cal_window = CalibrationWindow(widget=widget)
    template_window = TemplateWindow(widget=widget)
    signal_window = SignalWindow(widget=widget)
    progress_window = ProgressWindow(widget=widget)
    widget.addWidget(load_window)
    widget.addWidget(cal_window)
    widget.addWidget(template_window)
    widget.addWidget(signal_window)
    widget.addWidget(progress_window)
    widget.show()
    template_window.go_to_signal_window_signal.connect(signal_window.signal_window_is_being_shown)
    load_window.move_to_cal_window_signal.connect(cal_window.moved_to_cal_window)
    cal_window.go_to_template_window_signal.connect(template_window.load_processed_images_to_window)
    load_window.current_video_filepath_signal.connect(cal_window.get_current_video_filepath)
    load_window.current_video_filepath_signal.connect(template_window.get_current_video_filepath)
    load_window.current_video_filepath_signal.connect(signal_window.get_current_video_filepath)
    signal_window.start_processor_signal.connect(progress_window.start_processing)
    load_window.start_processing_signal.connect(progress_window.start_processing)
    progress_window.finished_processing_signal.connect(progress_window.finished)
    cal_window.loaded_new_video_signal.connect(signal_window.update_ui_from_new_video_loaded)

    app.exec_()

#: Main entry point
# if __name__ == "__main__":
#     start()
