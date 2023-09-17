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


class user_interfaceApp(App):
    def build(self):
        # screen_manager = ScreenManager()
        # homepage = BackgroundLayout()
        # screen_manager.add_widget(homepage)

        return BackgroundLayout()


class BackgroundLayout(Widget):
    def select_video(self):
        filepath: str = filedialog.askopenfilename()
        processor = video_processor.Processor()
        processor.run(video_filepath=filepath)


#: Functions
def main() -> None:
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')
    user_interfaceApp().run()


#: Main entry point
if __name__ == "__main__":
    main()
