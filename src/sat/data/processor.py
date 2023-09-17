# !/usr/bin/env python3
"""
This module is responsible for managing decoding of video for the application.
"""
import json
#: Imports
import logging
import os
from io import TextIOWrapper

import cv2
import numpy
from typing import List, Tuple, TextIO
from type import get_datetime_heading
from numpy import ndarray

#: Globals
_PROCESSING_METHODS = [cv2.TM_CCOEFF]


#: Class(es)
class Processor(object):
    """
    Class responsible for scanning spectrum analyzer video.
    """
    def __init__(self, video_fp: str, template_fp: str, trace_rgb: List[int], log_directory: str):
        self.video_fp = video_fp
        self.template_fp = template_fp
        self.trace_rgb = trace_rgb
        self.log_dir = log_directory
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self) -> None:
        logging.info(f"Attempting to load video from filepath {self.video_fp}...")
        template = self.load_template_im(template_img_fp=self.template_fp)
        capture = cv2.VideoCapture(self.video_fp)
        reading: bool = True
        log_filename: str = f'{self.log_dir}\\{get_datetime_heading()}_img_coordinates.csv'
        filestream = open(log_filename, 'a+')
        filestream.write(f"Log Filename: {log_filename}\nVideo Filepath: {video_fp}\n"
                         f"Trace RGB Selected: {self.trace_rgb}\nTemplate Filepath: {self.template_fp}\n\n"
                         f"Frame Number: \t (X, Y):\n")
        frame_counter = 0
        while reading:
            reading, frame = capture.read()
            if reading:
                frame_counter += 1
                self.process_frame(frame=frame, template=template, filestream=filestream,
                                   curr_frame_index=frame_counter)

    def process_frame(self, frame: ndarray, template: ndarray, filestream: TextIO, curr_frame_index: int) -> None:
        frame_grayscale = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        template_grayscale = cv2.cvtColor(template, cv2.IMREAD_GRAYSCALE)
        height_template, width_template, size = template_grayscale.shape
        reference_img = frame_grayscale.copy()
        cropped_img = self.crop_template_from_frame(reference_frame=reference_img,
                                                    template=template_grayscale, template_width=width_template,
                                                    template_height=height_template)
        self.find_datapoints_from_frame(frame=cropped_img, filestream=filestream, curr_frame_index=curr_frame_index)
        cv2.destroyAllWindows()

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

    @staticmethod
    def find_datapoints_from_frame(frame: ndarray, filestream: TextIO, curr_frame_index: int):
        bgr_lower = numpy.array([150, 200, 0, 255])
        bgr_high = numpy.array([255, 255, 10, 255])
        mask = cv2.inRange(frame, bgr_lower, bgr_high)
        height = len(frame)
        width = len(frame[0])
        coord = cv2.findNonZero(mask)
        normalized_coordinates: List[Tuple[float, float]] = []
        if len(coord) > 0:
            for i in range(len(coord)):
                for k in range(len(coord[i])):
                    curr_x_normalized = coord[i][k][0] / width
                    curr_y_normalized = coord[i][k][1] / height
                    normalized_element = (curr_x_normalized, curr_y_normalized)
                    normalized_coordinates.append(normalized_element)

        filestream.write(f"{curr_frame_index}, {','.join(map(str, normalized_coordinates))}\n")

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
    video_fp: str = f"C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\videos\\CW signal.mp4"
    template_fp: str = 'C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\imgs\\templates\\cw_signal_cutout2.png'
    log_directory: str = 'C:\\Users\\snipe\\OneDrive\\Desktop\\SAT'
    processor = Processor(video_fp=video_fp, template_fp=template_fp, trace_rgb=[4, 246, 255],
                          log_directory=log_directory)
    processor.run()
