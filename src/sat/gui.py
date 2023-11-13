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
    QTextEdit, QVBoxLayout, QColorDialog, QLCDNumber, QSlider, QToolButton, QCheckBox
from PyQt5 import uic
from processor import get_video_config, cv2, get_frame_count, get_specific_frame, get_reference_frame, \
    crop_template_from_frame, parse_trace_from_frame, load_template_im, show_frame
from distribution import __app_name__, __app_version__

#: Globals
CONFIG_FILENAME: str = 'config.ini'
DEFAULT_APP_WIDTH: int = 1277
DEFAULT_APP_HEIGHT: int = 830


class MainWindow(QMainWindow):
    current_video_filepath_signal: pyqtSignal = pyqtSignal(str)
    current_csv_filepath_signal: pyqtSignal = pyqtSignal(str)
    move_to_cal_window_signal: pyqtSignal = pyqtSignal(bool)

    def __init__(self, widget: QtWidgets.QStackedWidget):
        super(MainWindow, self).__init__()
        uic.loadUi("load_page.ui", self)
        self.setWindowTitle(f"Spectrum Analyzer Tool")
        self.current_loaded_video_filepath: str = ""
        self.widget = widget
        self.load_video_btn = self.findChild(QPushButton, "loadVidBtn")
        self.load_video_btn.clicked.connect(self.load_video_btn_callback)
        self.start_processor_preset_btn = self.findChild(QToolButton, "startProcessorPresets")
        self.start_processor_preset_btn.setStyleSheet("background-color : #D0D0D0")
        self.start_processor_preset_btn.clicked.connect(self.start_processor_preset_btn_callback)
        self.calibrate_processor_preset_btn = self.findChild(QToolButton, "calibrateProcessorPresets")
        self.calibrate_processor_preset_btn.setStyleSheet("background-color : #D0D0D0")
        self.calibrate_processor_preset_btn.clicked.connect(self.calibrate_processor_preset_btn_callback)
        self.calibrate_processor_preset_btn.setDisabled(True)
        self.start_processor_preset_btn.setDisabled(True)
        self.csv_textbox = self.findChild(QTextEdit, "csvFilepathLabel")
        self.ini_file_exists: bool = self.load_ini_file_to_app()
        self.setup_window_backgroud()

    def start_processor_preset_btn_callback(self) -> None:
        start: bool = confirm(msg=f"Are you sure you want to start processing?")
        if start:
            run_processor()

    def calibrate_processor_preset_btn_callback(self) -> None:
        self.go_to_calibration_window(filepath=self.current_loaded_video_filepath)

    def setup_window_backgroud(self) -> None:
        background_width = 800
        background_height = 600
        self.setFixedWidth(background_width)
        self.setFixedHeight(background_height)
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
                self.calibrate_processor_preset_btn.setStyleSheet("background-color : #FFFFDD")
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

    def load_ini_file_to_app(self) -> bool:
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.csv_textbox.setText(config['app']['csv_output_directory'])
        except Exception as e:
            logging.info(e)
            return False


class CalibrationWindow(QMainWindow):
    go_to_template_window_signal = pyqtSignal(bool)

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
            self.setFixedWidth(DEFAULT_APP_WIDTH)
            self.setFixedHeight(DEFAULT_APP_HEIGHT)
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

        self.start_processor_btn = self.findChild(QPushButton, 'startProcessingBtn')
        self.start_processor_btn.clicked.connect(self.start_processor_btn_callback)
        self.go_back_btn = self.findChild(QPushButton, 'goBackBtn')
        self.go_back_btn.clicked.connect(self.go_back_btn_callback)
        self.scan_text_slider: QSlider = self.findChild(QSlider, "selectTextScanSlider")
        self.scan_text_slider.valueChanged.connect(self.scan_text_slider_callback)
        self.record_freq_dbm_checkbox = self.findChild(QCheckBox, "recordFreqDbm")
        self.record_max_ampl_checkbox = self.findChild(QCheckBox, "recordMaxAmpl")
        self.record_relative_ampl_checkbox = self.findChild(QCheckBox, "recordRelativeSignal")
        self.record_freq_dbm_checkbox.clicked.connect(self.record_freq_dbm_checkbox_callback)
        self.record_max_ampl_checkbox.clicked.connect(self.record_max_ampl_checkbox_callback)
        self.record_relative_ampl_checkbox.clicked.connect(self.record_relative_ampl_checkbox_callback)
        self.select_csv_filepath = self.findChild(QPushButton, "csvFilepathBtn")
        self.select_csv_filepath.clicked.connect(self.select_csv_filepath_callback)
        self.select_csv_filepath_text = self.findChild(QTextEdit, "csvFilepathLabel")

        # Signal IDs
        self.center_freq_id = self.findChild(QTextEdit, "centerFreqID")
        self.span_freq_id = self.findChild(QTextEdit, "spanFreqID")
        self.ref_level_id = self.findChild(QTextEdit, "ref_level_ID")
        self.division_slider: QSlider = self.findChild(QSlider, "divisionSlider")
        self.division_slider.sliderReleased.connect(self.division_slider_callback)
        self.division_lcd = self.findChild(QLCDNumber, "divisionLcd")
        self.threshold_lcd = self.findChild(QLCDNumber, "thresholdLcd")
        self.scan_threshold_checkbox = self.findChild(QCheckBox, "scanThreshold")
        self.scan_threshold_checkbox.clicked.connect(self.scan_threshold_checkbox_callback)
        self.scan_threshold_slider: QSlider = self.findChild(QSlider, "thresholdSlider")
        self.scan_threshold_slider.sliderReleased.connect(self.scan_threshold_slider_callback)
        self.init_sliders()
        self.initialize_signal_window()

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
        try:
            with open(f"{CONFIG_FILENAME}") as config_file:
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self.division_slider.setValue(int(config['cal.signal']['total_db_divisions']))
                self.division_lcd.display(int(config['cal.signal']['total_db_divisions']))
                self.scan_threshold_slider.setValue(int(config['cal.signal']['threshold_percent']))
                self.threshold_lcd.display(int(config['cal.signal']['threshold_percent']))
                total_frames = int(config['cal.trace']['total_frames'])
                self.scan_text_slider.setMinimum(0)
                self.scan_text_slider.setMaximum(total_frames)
        except Exception as e:
            logging.info(f"Error while initializing trace rgb slider: {e}")

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
        csv_directory = str(QFileDialog.getExistingDirectory(self, "Select Directory to save CSV data to:"))
        logging.info(f"Location to save CSV file has been selected: {csv_directory}")
        self.select_csv_filepath_text.setText(csv_directory)
        try:
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
            conf.set("cal.signal", "scan_relative_amplitude_threshold", str(val))
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
        pass

    def go_back_btn_callback(self) -> None:
        self.widget.setCurrentIndex(2)

    def start_processor_btn_callback(self) -> None:
        pass

    def initialize_signal_window(self) -> None:

        try:
            conf = configparser.ConfigParser()
            conf.read_file(open(CONFIG_FILENAME, 'r'))
            center_freq_text = str(conf['cal.signal']['center_freq_text'])
            reference_level_text = str(conf['cal.signal']['reference_level_text'])
            span_text = str(conf['cal.signal']['span_text'])
            total_db_divisions = int(conf['cal.signal']['total_db_divisions'])
            csv_output_directory = str(conf['cal.signal']['csv_output_directory'])
            record_entire_signal_with_scaling_factors = (
                int(conf['cal.signal']['record_entire_signal_with_scaling_factors']))
            record_only_max_signal_with_scaling_factors = (
                int(conf['cal.signal']['record_entire_signal_with_scaling_factors']))
            record_entire_signal_with_no_scaling_factors = (
                int(conf['cal.signal']['record_entire_signal_with_scaling_factors']))
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

            self.select_csv_filepath_text.setText(csv_output_directory)
            self.center_freq_id.setText(center_freq_text)
            self.span_freq_id.setText(span_text)
            self.ref_level_id.setText(reference_level_text)
            self.division_slider.setValue(total_db_divisions)

            self.start_processor_btn.setStyleSheet("background-color : #62FFAD")

            # self.center_freq_id = self.findChild(QTextEdit, "centerFreqID")
            # # self.center_freq_id.textChanged.connect(self.center_freq_id_callback)
            #
            # self.span_freq_id = self.findChild(QTextEdit, "spanFreqID")
            # self.ref_level_id = self.findChild(QTextEdit, "ref_level_ID")
            # self.division_slider = self.findChild(QSlider, "divisionSlider")
            # self.division_lcd = self.findChild(QLCDNumber, "divisionLcd")
            # [cal.signal]
            # center_freq_text = CENTER
            # reference_level_text = RF
            # span_text = SPAN
            # total_db_divisions = 10
            # csv_output_directory = C: / Users / snipe / OneDrive / Desktop / SAT
            # record_entire_signal_with_scaling_factors = 1
            # record_only_max_signal_with_scaling_factors = 1
            # record_entire_signal_with_no_scaling_factors = 1
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
            self.start_processing()

    def start_processing(self) -> None:
        pass


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


def run_processor() -> None:
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
    widget.show()
    load_window.move_to_cal_window_signal.connect(cal_window.moved_to_cal_window)
    cal_window.go_to_template_window_signal.connect(template_window.load_processed_images_to_window)
    load_window.current_video_filepath_signal.connect(cal_window.get_current_video_filepath)
    load_window.current_video_filepath_signal.connect(template_window.get_current_video_filepath)
    load_window.current_video_filepath_signal.connect(signal_window.get_current_video_filepath)
    # load_window.current_csv_filepath.connect(cal_window.get_current_csv_filepath)
    app.exec_()


#: Main entry point
if __name__ == "__main__":
    start()
