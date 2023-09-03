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
        logging.info(f"Video from {video_filepath} has been loaded. {len(frames_found)} frames were read from the video.")
        self.show_frame(frames=frames_found)
        return frames_found

    def show_frame(self, frames:  list[ndarray]) -> None:
        cv2.imshow("video", frames[0])
        cv2.waitKey(0)
        hsv_color_spectrum = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)


class VideoProcessor(Decoder):
    """
    Class responsible for scanning spectrum analyzer video.
    """
    def __init__(self):
        super().__init__()

    def find_graph(self, frames:  list[ndarray], video_filepath: str) -> None:
        frames: numpy.ndarray = self.load_video(video_filepath=video_filepath)



#: Function(s)


#: Main entry point
if __name__ == "__main__":
    # Debugging
    _LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')
    temp_filepath: str = f"{os.pardir}/assets/video_specAn_1.mkv"
    # decoder = Decoder()
    # frames = decoder.load_video(video_filepath=temp_filepath)
    # decoder.show_frame(frames=frames)
