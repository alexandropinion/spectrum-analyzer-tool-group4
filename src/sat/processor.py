# !/usr/bin/env python3
"""
This module is responsible for managing decoding of video for the application.
"""

#: Imports
import logging
import os
import plotly.graph_objects as go
import cv2
import numpy
import pytesseract
import type
import re
from cv2 import Mat
from typing import List, Tuple, TextIO, Any, Optional
from numpy import ndarray, dtype, generic
from datetime import timedelta

#: Globals
_PROCESSING_METHODS = [cv2.TM_CCOEFF]  # Template Matching Correlation Coefficient
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different
UNIT_OPTIONS: List[str] = ['ghz', 'mhz', 'khz']


#: Class(es)
class Processor(object):
    """
    Class responsible for scanning spectrum analyzer video.
    """

    def __init__(self,
                 video_fp: str,
                 template_fp: str,
                 log_directory: str,
                 dbm_magnitude_threshold: float,
                 bgra_min_filter: List[float],
                 bgra_max_filter: List[float],
                 db_per_division: int,
                 center_freq_id: str = "CENTER",
                 ref_lvl_id: str = "RL",
                 span_lvl_id: str = "SPAN",
                 scan_for_relative_threshold: bool = False,
                 dbm_threshold: float = None,
                 record_relative_signal: bool = True,
                 record_scaled_signal: bool = True,
                 record_max_signal_scaled: bool = False,
                 default_threshold_graph_percentage: float = 0.8):
        self.db_per_division = db_per_division
        self.video_fp = video_fp
        self.template_fp = template_fp
        self.log_dir = log_directory
        self.dbm_thresh = dbm_magnitude_threshold
        self.bgra_min_filter = bgra_min_filter
        self.bgra_max_filter = bgra_max_filter
        self.scan_for_threshold = scan_for_relative_threshold
        self.dbm_threshold = dbm_threshold
        self.cf_tag = center_freq_id
        self.ref_lvl_tag = ref_lvl_id
        self.span_tag = span_lvl_id
        self.freq_units = ['ghz', 'mhz', 'khz']
        self.record_relative_signal = record_relative_signal
        self.record_scaled_signal = record_scaled_signal
        self.record_max_signal_scaled = record_max_signal_scaled
        self.fps: float = None
        self.read_first_scaled_factors: bool = False
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self, result: List[str]) -> None:
        logging.info(f"Attempting to load video from filepath {self.video_fp}...")
        template = load_template_im(template_img_fp=self.template_fp)
        capture = cv2.VideoCapture(self.video_fp)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        reading: bool = True
        heading = type.get_datetime_heading()
        test_cycle_dir: str = f"{self.log_dir}\\{heading}_cycle"
        os.makedirs(test_cycle_dir, exist_ok=True)
        relative_log_filename: str = ''
        max_signal_log_filename: str = ''
        scaled_signal_log_filename: str = ''
        filestream_relative_signal = None
        filestream_max_signal = None
        filestream_scaled_signal = None
        if self.record_relative_signal:
            relative_log_filename: str = f'{test_cycle_dir}\\{heading}_relative_signal_coordinates.csv'
            filestream_relative_signal = open(relative_log_filename, 'a+')
            filestream_relative_signal.write(f"Log Filename: {relative_log_filename}\n"
                                             f"Video Filepath: {self.video_fp}\n"
                                             f"Template Filepath: {self.template_fp}\n"
                                             f"Video FPS: {self.fps}\n"
                                             f"DBM Magnitude Limit (0 to 1): {self.dbm_thresh}\n"
                                             f"BGRA Minimum Trace Filter: {self.bgra_min_filter}\n"
                                             f"BGRA Maximum Trace Filter: {self.bgra_max_filter}\n\n\n"
                                             f"Time (H:M:S:mS), \t(Xpos, Ypos):\n")
        if self.record_max_signal_scaled:
            max_signal_log_filename: str = f'{test_cycle_dir}\\{heading}_max_signal_coordinates.csv'
            filestream_max_signal = open(max_signal_log_filename, 'a+')
            filestream_max_signal.write(f"Log Filename: {max_signal_log_filename}\n"
                                        f"Video Filepath: {self.video_fp}\n"
                                        f"Template Filepath: {self.template_fp}\n"
                                        f"Video FPS: {self.fps}\n"
                                        f"DBM Magnitude Limit (0 to 1): {self.dbm_thresh}\n"
                                        f"BGRA Minimum Trace Filter: {self.bgra_min_filter}\n"
                                        f"BGRA Maximum Trace Filter: {self.bgra_max_filter}\n\n\n"
                                        f"Time (H:M:S:mS),\t(Freq, dBm):\n")

        if self.record_scaled_signal:
            scaled_signal_log_filename: str = f'{test_cycle_dir}\\{heading}_scaled_signal_coordinates.csv'
            filestream_scaled_signal = open(scaled_signal_log_filename, 'a+')
            filestream_scaled_signal.write(f"Log Filename: {scaled_signal_log_filename}\n"
                                           f"Video Filepath: {self.video_fp}\n"
                                           f"Template Filepath: {self.template_fp}\n"
                                           f"Video FPS: {self.fps}\n"
                                           f"DBM Magnitude Limit (0 to 1): {self.dbm_thresh}\n"
                                           f"BGRA Minimum Trace Filter: {self.bgra_min_filter}\n"
                                           f"BGRA Maximum Trace Filter: {self.bgra_max_filter}\n\n\n"
                                           f"Time (H:M:S:mS),\t(Freq, dBm):\n")

        frame_counter: int = 0
        frame_thresholds: List[Tuple[bool, int]] = []
        while reading:
            reading, frame = capture.read()
            if reading:
                frame_counter += 1
                frame_threshold_found = (
                    self.process_frame_signal_failures(frame=frame,
                                                       template=template,
                                                       curr_frame_index=frame_counter,
                                                       dbm_threshold=self.dbm_thresh,
                                                       relative_signal_filestream=filestream_relative_signal,
                                                       max_signal_filestream=filestream_max_signal,
                                                       scaled_signal_filestream=filestream_scaled_signal,
                                                       record_scaled_coordinate=self.record_scaled_signal,
                                                       record_max_coordinate=self.record_max_signal_scaled,
                                                       record_relative_coordinate=self.record_relative_signal
                                                       ))
                frame_thresholds.append((frame_threshold_found, frame_counter))
        if self.scan_for_threshold and self.record_relative_signal:
            self.graph_dbm_thresholds(thresholds=frame_thresholds, total_frames=frame_counter, frames_per_sec=self.fps,
                                      video_fp=self.video_fp, data_logfile=relative_log_filename)

        if filestream_relative_signal is not None:
            try:
                filestream_relative_signal.close()
                result.append(relative_log_filename)
            except Exception:
                pass
        if filestream_max_signal is not None:
            try:
                filestream_max_signal.close()
                result.append(max_signal_log_filename)
            except Exception:
                pass
        if filestream_scaled_signal is not None:
            try:
                filestream_scaled_signal.close()
                result.append(scaled_signal_log_filename)
            except Exception:
                pass
        cv2.destroyAllWindows()

    def graph_dbm_thresholds(self, thresholds: List[Tuple[bool, int]], total_frames: int, frames_per_sec: float,
                             video_fp: str, data_logfile: str):
        elapsed_time_s: float = (total_frames / frames_per_sec)
        x_axis: List = []
        y_axis: List = []
        for i in range(len(thresholds)):
            x_axis.append(thresholds[i][1] / frames_per_sec)
            y_axis.append(int(thresholds[i][0]))
        fig = go.Figure(go.Scatter(x=x_axis, y=y_axis, mode='lines'))
        total_elapsed_datetime: timedelta = timedelta(seconds=elapsed_time_s)
        fig.update_layout(title=f"Corresponding Logfile: {data_logfile}, "
                                f"Total Elapsed Time: {total_elapsed_datetime}, "
                                f"Video Filepath: {video_fp}, "
                                f"Template Filepath: {self.template_fp}, "
                                f"Video FPS: {frames_per_sec}, "
                                f"DBM Magnitude Limit (0 to 1): {self.dbm_thresh}, "
                                f"BGRA Minimum Trace Filter: {self.bgra_min_filter}, "
                                f"BGRA Maximum Trace Filter: {self.bgra_max_filter}, ")
        fig.show()

    def process_frame_signal_failures(self, frame: ndarray,
                                      template: ndarray,
                                      curr_frame_index: int,
                                      dbm_threshold: float,
                                      relative_signal_filestream: Optional[TextIO | None] = None,
                                      max_signal_filestream: Optional[TextIO | None] = None,
                                      scaled_signal_filestream: Optional[TextIO | None] = None,
                                      record_scaled_coordinate: bool = False,
                                      record_max_coordinate: bool = False,
                                      record_relative_coordinate: bool = False
                                      ) -> bool:

        reference_img, height_template, width_template, size, template_grayscale = (
            get_reference_frame(frame=frame, template=template))

        cropped_img, inverse_cropped_img = crop_template_from_frame(reference_frame=reference_img,
                                                                    template=template_grayscale,
                                                                    template_width=width_template,
                                                                    template_height=height_template)

        under_threshold, coordinates = (
            self.scan_for_dbm_threshold(frame=cropped_img,
                                        curr_frame_index=curr_frame_index,
                                        bgra_max_limit=self.bgra_max_filter,
                                        bgra_min_limit=self.bgra_min_filter,
                                        dbm_threshold=dbm_threshold,
                                        relative_signal_filestream=relative_signal_filestream,
                                        max_signal_filestream=max_signal_filestream,
                                        scaled_signal_filestream=scaled_signal_filestream,
                                        record_max_coordinate=record_max_coordinate,
                                        record_scaled_coordinate=record_scaled_coordinate,
                                        record_relative_coordinate=record_relative_coordinate
                                        ))

        if record_max_coordinate or record_scaled_coordinate:
            successful, center_freq, reference_level, span_freq = (
                get_center_freq_and_ref_level(
                    frame=inverse_cropped_img, center_freq_tag=self.cf_tag,
                    reference_level_tag=self.ref_lvl_tag, span_tag=self.span_tag))
        return under_threshold

    def scan_for_dbm_threshold(self,
                               frame: ndarray,
                               curr_frame_index: int,
                               bgra_min_limit: List[float],
                               bgra_max_limit: List[float],
                               dbm_threshold: float,
                               relative_signal_filestream: Optional[TextIO | None] = None,
                               max_signal_filestream: Optional[TextIO | None] = None,
                               scaled_signal_filestream: Optional[TextIO | None] = None,
                               record_scaled_coordinate: bool = False,
                               record_max_coordinate: bool = False,
                               record_relative_coordinate: bool = False
                               ) -> Tuple[bool, list]:
        coordinates, under_threshold = (
            self.parse_datapoints_from_frame(frame=frame,
                                             curr_frame_index=curr_frame_index,
                                             bgra_min_limit=bgra_min_limit,
                                             bgra_max_limit=bgra_max_limit,
                                             dbm_threshold=dbm_threshold,
                                             relative_signal_filestream=relative_signal_filestream,
                                             max_signal_filestream=max_signal_filestream,
                                             scaled_signal_filestream=scaled_signal_filestream,
                                             record_max_coordinate=record_max_coordinate,
                                             record_scaled_coordinate=record_scaled_coordinate,
                                             record_relative_coordinate=record_relative_coordinate))
        return under_threshold, coordinates

    def parse_datapoints_from_frame(self, frame: ndarray, curr_frame_index: int,
                                    bgra_min_limit: List[float], bgra_max_limit: List[float],
                                    dbm_threshold: float,
                                    relative_signal_filestream: Optional[TextIO | None] = None,
                                    max_signal_filestream: Optional[TextIO | None] = None,
                                    scaled_signal_filestream: Optional[TextIO | None] = None,
                                    record_scaled_coordinate: bool = False,
                                    record_max_coordinate: bool = False,
                                    record_relative_coordinate: bool = False) -> Tuple[list, bool]:
        mask = parse_trace_from_frame(bgra_min_limit=bgra_min_limit, bgra_max_limit=bgra_max_limit, frame=frame)
        height = len(frame)
        width = len(frame[0])
        coord = cv2.findNonZero(mask)
        normalized_coordinates: List[Tuple[float, float]] = []
        under_threshold: bool = False
        if coord is not None:
            for i in range(len(coord)):
                for k in range(len(coord[i])):
                    curr_x_normalized = coord[i][k][0] / width
                    curr_y_normalized = coord[i][k][1] / height
                    normalized_element = (curr_x_normalized, curr_y_normalized)
                    normalized_coordinates.append(normalized_element)
                    if curr_y_normalized > dbm_threshold:  # coordinates start at top-left
                        under_threshold = True
        formatted_time = type.convert_seconds_to_datetime_hour_min_sec_ms(seconds=curr_frame_index / self.fps)
        try:
            if relative_signal_filestream is not None and record_relative_coordinate is True:
                relative_signal_filestream.write(f"{formatted_time}, {','.join(map(str, normalized_coordinates))}\n")
            if max_signal_filestream is not None and record_max_coordinate is True:
                max_signal_filestream.write(f"{formatted_time}, {','.join(map(str, normalized_coordinates))}\n")
            if scaled_signal_filestream is not None and record_scaled_coordinate is True:
                scaled_signal_filestream.write(f"{formatted_time}, {','.join(map(str, normalized_coordinates))}\n")
        except Exception as e:
            logging.info(f"Issue while writing to filestream: {e}")
        return normalized_coordinates, under_threshold


# Function(s):
def load_template_im(template_img_fp: str) -> cv2.imread:
    return cv2.imread(filename=template_img_fp, flags=0)


def show_frame(frame: ndarray, title: str) -> None:
    logging.debug(f"DEMO: Showing the following frame: {title}...")
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_trace_from_frame(bgra_min_limit: List[float], bgra_max_limit: List[float], frame: ndarray):
    bgr_lower = numpy.array(bgra_min_limit)
    bgr_high = numpy.array(bgra_max_limit)
    mask = cv2.inRange(frame, bgr_lower, bgr_high)
    return mask


def get_video_config(filepath: str) -> Tuple[float, bool, ndarray]:
    capture = cv2.VideoCapture(filepath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    read, frame = capture.read()
    return fps, read, frame


def get_frame_count(filepath: str) -> int:
    capture = cv2.VideoCapture(filepath)
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_specific_frame(filepath: str, frame_num: int) -> ndarray:
    try:
        capture = cv2.VideoCapture(filepath)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        res, frame = capture.read()
        return frame
    except Exception as e:
        logging.info(f"Error while trying to read frame number {frame_num} from video {filepath}: {e}")


def get_reference_frame(frame: ndarray, template: ndarray) -> tuple[Mat | ndarray | ndarray[
    Any, dtype[generic | generic]], int, int, int, Mat | ndarray | ndarray[Any, dtype[generic | generic]]] | None:
    try:
        frame_grayscale = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        template_grayscale = cv2.cvtColor(template, cv2.IMREAD_GRAYSCALE)

        height_template, width_template, size = template_grayscale.shape
        reference_img = frame_grayscale.copy()
        return reference_img, height_template, width_template, size, template_grayscale
    except Exception as e:
        logging.info(f"Error while capturing reference frame: {e}")
        return None


def crop_template_from_frame(reference_frame: ndarray, template: ndarray,
                             template_width: int, template_height: int) -> Tuple[ndarray, ndarray]:
    ref_copy = reference_frame.copy()
    result = cv2.matchTemplate(reference_frame, template, _PROCESSING_METHODS[0])
    min_val, max_val, min_location, max_location = cv2.minMaxLoc(result)
    if _PROCESSING_METHODS in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        img_location = min_location
    else:
        img_location = max_location
    bottom_right = (img_location[0] + template_width, img_location[1] + template_height)
    cv2.rectangle(reference_frame, img_location, bottom_right, 255, 5)
    inverse_cropped_img = cv2.rectangle(ref_copy, img_location, bottom_right, (255, 255, 255), -1)

    x1, y1 = img_location
    x2, y2 = bottom_right
    cropped_img = reference_frame[y1:y2, x1:x2]
    return cropped_img, inverse_cropped_img


def draw_boxes_around_text_in_frame(frame: ndarray, text_to_find: List[str]) -> ndarray:
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    boxes_threshold_img = threshold_img.copy()
    text_dataframe = pytesseract.image_to_data(boxes_threshold_img, lang="eng", config="--psm 6",
                                               output_type=pytesseract.Output.DATAFRAME)
    frame_to_draw_on = frame.copy()
    for line_num, words_per_line in text_dataframe.groupby("line_num"):
        words_per_line = words_per_line[words_per_line["conf"] >= 5]
        if not len(words_per_line):
            continue

        words = words_per_line["text"].values
        line = " ".join(words)

        for each_text in text_to_find:
            if each_text.lower() in line.lower():
                word_boxes = []
                for left, top, width, height in words_per_line[["left", "top", "width", "height"]].values:
                    word_boxes.append((left, top))
                    word_boxes.append((left + width, top + height))
                x, y, w, h = cv2.boundingRect(numpy.array(word_boxes))
                frame_to_draw_on = cv2.rectangle(
                    frame_to_draw_on, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=3)
    return frame_to_draw_on


def get_center_freq_and_ref_level(
        frame: ndarray, center_freq_tag: str, reference_level_tag: str, span_tag: str) \
        -> tuple[
               bool, float, float, float] | \
           tuple[bool, float, float]:
    frame_copy = frame.copy()

    # Normalize all of the tags
    center_freq_tag_lower = center_freq_tag.lower()
    reference_level_tag_lower = reference_level_tag.lower()
    span_freq_tag_lower = span_tag.lower()

    # Preprocess data for text detection
    gray_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    _, thresholded_image = cv2.threshold(blurred_image, 97, 255, cv2.THRESH_BINARY_INV)

    # Detect all text discovered in frame
    get_all_text: str = pytesseract.image_to_string(thresholded_image, lang='eng', config='--psm 6')
    result: str = ''
    for char in get_all_text:
        if char.isalpha():
            result += char.lower()
        else:
            result += char
    # print(f"result text = {result}")
    center_freq_position = result.find(center_freq_tag_lower)
    rf_level_position = result.find(reference_level_tag_lower)
    span_level_position = result.find(span_freq_tag_lower)

    found_values: bool = False
    if center_freq_position != -1 and rf_level_position != -1 and span_level_position != -1:

        cf_unit = 'ghz'
        cf_str = parsed_signal_from_str(
            string=result, text_position=center_freq_position, tag=center_freq_tag_lower, unit=cf_unit)
        if cf_str is None: # TODO: Add similar IDs that are common misinterpretations
            cf_unit = 'gh2'
            cf_str = parsed_signal_from_str(
                string=result, text_position=center_freq_position, tag=center_freq_tag_lower, unit=cf_unit)
        if cf_str is None:
            cf_unit = 'ghe'
            cf_str = parsed_signal_from_str(
                string=result, text_position=center_freq_position, tag=center_freq_tag_lower, unit=cf_unit)
        if cf_str is None:
            cf_unit = 'mhz'
            cf_str = parsed_signal_from_str(
                string=result, text_position=center_freq_position, tag=center_freq_tag_lower, unit=cf_unit)
        if cf_str is None:
            cf_unit = 'khz'
            cf_str = parsed_signal_from_str(
                string=result, text_position=center_freq_position, tag=center_freq_tag_lower, unit=cf_unit)

        span_unit = 'mhz'
        span_str = parsed_signal_from_str(
            string=result, text_position=span_level_position, tag=span_freq_tag_lower, unit=span_unit)
        if span_str is None:
            span_unit = 'ghz'
            span_str = parsed_signal_from_str(
                string=result, text_position=span_level_position, tag=span_freq_tag_lower, unit=span_unit)
        if span_str is None:
            span_unit = 'khz'
            span_str = parsed_signal_from_str(
                string=result, text_position=span_level_position, tag=span_freq_tag_lower, unit=span_unit)

        parsed_ref_str = parsed_signal_from_str(
            string=result, text_position=rf_level_position, tag=reference_level_tag_lower, unit='dbm')

        if parsed_ref_str is not None and span_str is not None and cf_str is not None:
            try:
                #conditioned_rf_str = parsed_ref_str.replace('o', '0')
                conditioned_cf_str = condition_cf_str(cf_str=cf_str)
                conditioned_span_str = condition_span_string(span_str=span_str)
                conditioned_rf_str = condition_rf_string(rf_str=parsed_ref_str)
                # print(f"conditioned_rf_str = {conditioned_rf_str}\n"
                #       f"conditioned_span_str = {conditioned_span_str}\n"
                #       f"conditioned_cf_str = {conditioned_cf_str}")
                multiplier_span = find_freq_multiplier_from_unit_str(string=span_unit)
                multiplier_cf = find_freq_multiplier_from_unit_str(string=cf_unit)
                ref_lvl = float(conditioned_rf_str)
                center_freq = float(conditioned_cf_str) * multiplier_cf
                span_freq = float(conditioned_span_str) * multiplier_span
                if center_freq == 0.0 or span_freq == 0.0:
                    found_values = False
                else:
                    found_values = True
                print(f"ref lvl = {ref_lvl}\ncenter freq = {center_freq}\nspan freq = {span_freq}\n"
                      f"ref lvl unit = dbm\ncenter freq unit = {cf_unit}\nspan freq unit = {span_unit}")
                return found_values, ref_lvl, center_freq, span_freq
            except Exception as e:
                logging.debug(f"Could not convert center freq and ref lvl to float: {e}")
        else:
            found_values = False
    return found_values, 0.0, 0.0, 0.0


def condition_cf_str(cf_str: str) -> str:
    conditioned_cf_str = cf_str.replace(',', '.')
    conditioned_cf_str = conditioned_cf_str.replace('o', '0')
    conditioned_cf_str = conditioned_cf_str.replace('q', '0')
    conditioned_cf_str = conditioned_cf_str.replace('@', '0')
    letters = re.compile(r'[a-zA-Z]')  # Match any letter (case-insensitive)
    result_string = re.sub(letters, '', conditioned_cf_str)
    result_string = result_string + '0'
    result_string = result_string.replace(' ', '')
    return result_string

def condition_rf_string(rf_str: str) -> str:
    condition_rf_string = rf_str.replace('_', '')
    condition_rf_string = condition_rf_string.replace('o', '0')
    condition_rf_string = condition_rf_string.replace('l', '')
    return condition_rf_string

def condition_span_string(span_str: str) -> str:
    condition_span_string = span_str.replace(',', '.')
    condition_span_string = condition_span_string.replace('q', '0')
    letters = re.compile(r'[a-zA-Z]')  # Match any letter (case-insensitive)
    result_string = re.sub(letters, '', condition_span_string)
    result_string = result_string + '0'
    result_string = result_string.replace(' ', '')
    return result_string

def find_freq_multiplier_from_unit_str(string: str) -> float:
    formatted_string = string.lower()
    multiplier: float = 1.0
    if formatted_string == 'ghz' or formatted_string == 'gh2' or formatted_string == 'ghe':
        multiplier = 1e9
    elif formatted_string == 'mhz':
        multiplier = 1e6
    elif formatted_string == 'khz':
        multiplier = 1e3
    else:
        multiplier = 1.0
    return multiplier


def parsed_signal_from_str(string: str, text_position: int, tag: str, unit: str) -> str:
    text_to_parse: str = string
    final_str: str = None
    try:
        start_str = text_to_parse[text_position + len(tag):].strip()
        stop_index = start_str.index(unit)
        final_str = start_str[0:stop_index]
    except ValueError as e:
        logging.debug(f"Signal parse result: {e}")
        final_str = None
    return final_str


def get_graph_value_str(text: str, tag: str, units: List[str], text_position: int) -> Tuple[str, str] | None:
    val_after_tag = text[text_position + len(tag):].strip()

    parsed_strings = []
    for k, unit in enumerate(units):
        try:
            parsed = val_after_tag[val_after_tag.find(tag) + len(tag):val_after_tag.rfind(unit)]
            # print(f"parsed = {parsed}")
            if len(parsed) > 1:
                parsed_strings.append(parsed)
        except Exception as e:
            logging.info(f"No substring found: {e}")
    if len(parsed_strings) == 0:
        return None
    parsed_string = str(min(parsed_strings, key=len))
    index_min = min(range(len(parsed_strings)), key=parsed_strings.__getitem__)
    final_parsed_str = f"{parsed_string} {units[index_min]}"
    return final_parsed_str, units[index_min]


#: Main entry point
if __name__ == "__main__":
    # FOR DEMO
    _LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.CRITICAL,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')

    # FOR DEMO
    video_fp: str = f"C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\videos\\CW signal.mp4"
    template_fp: str = 'C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\src\\sat\\imgs\\template_1.png'
    log_directory: str = 'C:\\Users\\snipe\\OneDrive\\Desktop\\SAT'
    processor = Processor(video_fp=video_fp,
                          template_fp=template_fp,
                          log_directory=log_directory,
                          dbm_magnitude_threshold=0.8,
                          bgra_min_filter=[180, 187, 0, 255],
                          bgra_max_filter=[255, 255, 10, 255],
                          db_per_division=10,
                          center_freq_id='CENTER',
                          span_lvl_id='SPAN',
                          ref_lvl_id='RL')
    filepath_to_save_csv = []
    processor.run(result=filepath_to_save_csv)
    print(f"filepath after run: {filepath_to_save_csv[0]}")
