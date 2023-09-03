# !/usr/bin/env python3
"""
This module is responsible for managing user interactions with the application.
"""
from kivy.lang import Builder

#: Imports
import video_processor
import tkinter
from tkinter import filedialog
import logging
import kivy
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget

#: Globals
_LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
Builder.load_file('user_interface.kv')


#: Classes
class User(object):
    def __init__(self) -> None:
        logging.basicConfig(level=logging.DEBUG,
                            format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')
        self.decoder: video_processor.Decoder = video_processor.Decoder()

    def select_video(self) -> str:
        video_filepath: str = tkinter.filedialog.askopenfilename(filetypes=[("mp4", ".mp4"), ("mkv", ".mkv")])
        logging.info(f"User selected filepath for video is: {video_filepath}")
        return video_filepath

    def process_video(self, filepath: str) -> None:
        self.decoder.load_video(video_filepath=filepath)


class user_interfaceApp(App):
    def build(self):
        # screen_manager = ScreenManager()
        # homepage = BackgroundLayout()
        # screen_manager.add_widget(homepage)

        return BackgroundLayout()


class BackgroundLayout(Widget):
    def temp_func(self):
        pass


#: Functions
def main() -> None:
    user_interfaceApp().run()
    # user = User()
    # video_filepath = user.select_video()
    # user.process_video(filepath=video_filepath)


#: Main entry point
if __name__ == "__main__":
    main()
