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
from typing import List, Tuple, TextIO
from type import get_datetime_heading
from numpy import ndarray
from datetime import timedelta

#: Globals
_PROCESSING_METHODS = [cv2.TM_CCOEFF]


#: Class(es)
class Processor(object):
    """
    Class responsible for scanning spectrum analyzer video.
    """

    def __init__(self, video_fp: str, template_fp: str, log_directory: str,
                 dbm_magnitude_threshold: float, bgra_min_filter: List[float], bgra_max_filter: List[float]):
        self.video_fp = video_fp
        self.template_fp = template_fp
        self.log_dir = log_directory
        self.dbm_thresh = dbm_magnitude_threshold
        self.bgra_min_filter = bgra_min_filter
        self.bgra_max_filter = bgra_max_filter
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self) -> None:
        logging.info(f"Attempting to load video from filepath {self.video_fp}...")
        template = self.load_template_im(template_img_fp=self.template_fp)
        capture = cv2.VideoCapture(self.video_fp)
        fps = capture.get(cv2.CAP_PROP_FPS)
        reading: bool = True
        log_filename: str = f'{self.log_dir}\\{get_datetime_heading()}_img_coordinates.csv'
        filestream = open(log_filename, 'a+')
        filestream.write(f"Log Filename: {log_filename}\n"
                         f"Video Filepath: {video_fp}\n"
                         f"Template Filepath: {self.template_fp}\n"
                         f"Video FPS: {fps}\n"
                         f"DBM Magnitude Limit (0 to 1): {self.dbm_thresh}\n"
                         f"BGRA Minimum Trace Filter: {self.bgra_min_filter}\n"
                         f"BGRA Maximum Trace Filter: {self.bgra_max_filter}\n\n\n"
                         f"Frame Number:\t(X, Y):\n")
        frame_counter: int = 0
        frame_thresholds: List[Tuple[bool, int]] = []
        while reading:
            reading, frame = capture.read()
            if reading:
                frame_counter += 1
                frame_threshold_found = self.process_frame_signal_failures(frame=frame, template=template,
                                                                           filestream=filestream,
                                                                           curr_frame_index=frame_counter,
                                                                           dbm_threshold=self.dbm_thresh)
                frame_thresholds.append((frame_threshold_found, frame_counter))
        self.graph_dbm_thresholds(thresholds=frame_thresholds, total_frames=frame_counter, frames_per_sec=fps,
                                  video_fp=video_fp, data_logfile=log_filename)
        filestream.close()
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

    def process_frame_signal_failures(self, frame: ndarray, template: ndarray, filestream: TextIO,
                                      curr_frame_index: int,
                                      dbm_threshold: float) -> bool:
        frame_grayscale = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        template_grayscale = cv2.cvtColor(template, cv2.IMREAD_GRAYSCALE)
        height_template, width_template, size = template_grayscale.shape
        reference_img = frame_grayscale.copy()
        cropped_img = self.crop_template_from_frame(reference_frame=reference_img,
                                                    template=template_grayscale, template_width=width_template,
                                                    template_height=height_template)
        under_threshold = self.scan_for_dbm_threshold(frame=cropped_img, filestream=filestream,
                                                      curr_frame_index=curr_frame_index,
                                                      bgra_max_limit=self.bgra_max_filter,
                                                      bgra_min_limit=self.bgra_min_filter,
                                                      dbm_threshold=dbm_threshold)
        return under_threshold

    @staticmethod
    def crop_template_from_frame(reference_frame: ndarray, template: ndarray,
                                 template_width: int, template_height: int) -> ndarray:
        result = cv2.matchTemplate(reference_frame, template, _PROCESSING_METHODS[0])
        min_val, max_val, min_location, max_location = cv2.minMaxLoc(result)
        if _PROCESSING_METHODS in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            img_location = min_location
        else:
            img_location = max_location
        bottom_right = (img_location[0] + template_width, img_location[1] + template_height)
        cv2.rectangle(reference_frame, img_location, bottom_right, 255, 5)
        x1, y1 = img_location
        x2, y2 = bottom_right
        cropped_img = reference_frame[y1:y2, x1:x2]
        return cropped_img

    def scan_for_dbm_threshold(self, frame: ndarray, filestream: TextIO, curr_frame_index: int,
                               bgra_min_limit: List[float], bgra_max_limit: List[float],
                               dbm_threshold: float) -> bool:
        coordinates, under_threshold = self.parse_datapoints_from_frame(frame=frame, filestream=filestream,
                                                                        curr_frame_index=curr_frame_index,
                                                                        bgra_min_limit=bgra_min_limit,
                                                                        bgra_max_limit=bgra_max_limit,
                                                                        dbm_threshold=dbm_threshold)
        return under_threshold

    @staticmethod
    def parse_datapoints_from_frame(frame: ndarray, filestream: TextIO, curr_frame_index: int,
                                    bgra_min_limit: List[float], bgra_max_limit: List[float],
                                    dbm_threshold: float) -> Tuple[list, bool]:
        bgr_lower = numpy.array(bgra_min_limit)
        bgr_high = numpy.array(bgra_max_limit)
        mask = cv2.inRange(frame, bgr_lower, bgr_high)
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
        filestream.write(f"{curr_frame_index}, {','.join(map(str, normalized_coordinates))}\n")
        return normalized_coordinates, under_threshold

    @staticmethod
    def load_template_im(template_img_fp: str) -> cv2.imread:
        return cv2.imread(filename=template_img_fp, flags=0)


# Function(s):
def show_frame(frame: ndarray) -> None:
    cv2.imshow('Detected', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#: Main entry point
if __name__ == "__main__":
    # Debugging
    _LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')

    # Debugging
    video_fp: str = f"C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\videos\\CW signal.mp4"
    template_fp: str = 'C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\imgs\\templates\\cw_signal_cutout2.png'
    log_directory: str = 'C:\\Users\\snipe\\OneDrive\\Desktop\\SAT'
    processor = Processor(video_fp=video_fp, template_fp=template_fp,
                          log_directory=log_directory, dbm_magnitude_threshold=0.8,
                          bgra_min_filter=[150, 200, 0, 255],
                          bgra_max_filter=[255, 255, 10, 255])
    # processor.graph_dbm_thresholds(thresholds=[(True, 1), (False, 2), (True, 3), (False, 4), (True, 5)], total_frames=5,
    #                                frames_per_sec=30, video_fp="video_fp", data_logfile="datalog")
    processor.run()
