# !/usr/bin/env python3
"""
This module is responsible for managing decoding of video for the application.
"""
#: Imports
import logging
import os
import cv2
import numpy
from typing import List

from numpy import ndarray


#: Globals

#: Class(es)
class Decoder(object):
    """
    Class responsible for decoding video files.
    """

    def __init__(self) -> None:
        pass

    def load_video(self, video_filepath: str) -> list[ndarray]:
        logging.info(f"Attempting to load video from filepath {video_filepath}...")
        capture = cv2.VideoCapture(video_filepath)
        frames_found: list[ndarray] = []
        reading: bool = True
        while reading:
            reading, frame = capture.read()
            if reading:
                frames_found.append(frame)
        logging.info(
            f"Video from {video_filepath} has been loaded. {len(frames_found)} frames were read from the video.")
        show_frame(frames=frames_found)
        return frames_found


class VideoProcessor(Decoder):
    """
    Class responsible for scanning spectrum analyzer video.
    """

    def __init__(self):
        super().__init__()

    def find_graph(self, frames: list[ndarray], video_filepath: str) -> None:
        frames: list[ndarray] = self.load_video(video_filepath=video_filepath)


#: Function(s)
def show_frame(frames: list[ndarray]) -> None:

    for frame in frames[0:10]:
        im = frame
        im_new = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Detected', im_new)
        cv2.waitKey(0)

        template = cv2.imread('../../../assets/imgs/templates/grid.jpg', 0)



        template_new = cv2.cvtColor(template, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Detected', template_new)
        cv2.waitKey(0)
        im_copy = im_new.copy()
        height_template, width_template, size = template_new.shape
        print(f"height of template = {height_template}, width of template = {width_template}")
        methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                   cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]

        methods = [cv2.TM_SQDIFF]

        for each_method in methods:
            check_img = im_copy.copy()
            try:
                result = cv2.matchTemplate(check_img, template_new, each_method)
                min_val, max_val, min_location, max_location = cv2.minMaxLoc(result)
                print(f"{min_val},{max_val},{min_location},{max_location}")
                if each_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    loc = min_location
                else:
                    loc = max_location
                bot_right = (loc[0] + width_template, loc[1] + height_template)
                cv2.rectangle(im_copy, loc, bot_right, 255, 5)
                cv2.imshow('Detected', im_copy)
                cv2.waitKey(0)
            except Exception as err:
                logging.debug(f"Exception while matching template = {err}")

        cv2.destroyAllWindows()


#: Main entry point
if __name__ == "__main__":
    # Debugging
    _LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')
    # temp_filepath: str = f"{os.pardir}/assets/video_specAn_1.mkv"
    temp_filepath: str = f"{os.pardir}/assets/CW signal.mp4"
    # temp_filepath: str = f"{os.pardir}/assets/Pulsed Signal.mp4"
    decoder = Decoder()
    frames_read = decoder.load_video(video_filepath=temp_filepath)
