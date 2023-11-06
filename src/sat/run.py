#!/usr/bin/env python3

#: Imports
import logging
import time
from datetime import datetime
from threading import Thread
from typing import List

import processor as proc
import tkinter
import os
from time import sleep
from tkinter import filedialog, messagebox, ttk

from gui import start

#: Globals
root = tkinter.Tk()
root.title("Spectrum Analyzer Tool: Demo - Group 4 - Presentation 2 (Chris, George, Jeffrey, Seema, Alex)")
root.withdraw()


def get_csv_filepath() -> str:
    p = ttk.Progressbar(root, orient="horizontal", length=1000, mode="determinate",
                        takefocus=True, maximum=100)
    p['value'] = 0
    p.pack(ipady=30)

    video_fp: str = filedialog.askopenfilename(filetypes=[('.mkv', '.mp4')], initialdir=os.getcwd(),
                                               title=f"Spectrum Analyzer Tool: Select Video to Load...")
    log_directory: str = filedialog.askdirectory(initialdir=os.getcwd(),
                                                 title=f"Spectrum Analyzer Tool: Select directory to save CSV results...")
    template_fp: str = 'C:\\Users\\snipe\\OneDrive\\Documents\\GitHub\\spectrum-analyzer-tool-group5\\assets\\imgs\\templates\\template_1.png'

    processor = proc.Processor(video_fp=video_fp, template_fp=template_fp,
                               log_directory=log_directory, dbm_magnitude_threshold=0.8,
                               bgra_min_filter=[150, 200, 0, 255],
                               bgra_max_filter=[255, 255, 10, 255])

    root.deiconify()
    result: List[str] = []
    processing_thread = Thread(target=processor.run, args=(result,))
    processing_thread.daemon = False
    processing_thread.start()

    def loading(thread: Thread):
        try:
            now = datetime.now()
            if p['value'] < 100:
                p['value'] += 1
                root.after(1000, loading, processing_thread)
            if thread.is_alive():
                print(f"{now}: Thread is running!")
            else:
                print(f"{now}: Thread is dead!!")
                root.destroy()
        except Exception:
            pass

    root.after(1000, loading, processing_thread)
    root.wm_attributes('-toolwindow', 'True')
    root.mainloop()
    return result[0]


#: Main entry point
def main() -> None:
    # FOR DEMO
    _LOG_FORMAT = "[%(asctime)s : %(filename)s->%(funcName)s():%(lineno)s] : %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=_LOG_FORMAT, datefmt='%d-%b-%y %H:%M:%S')
    title: str = "Spectrum Analyzer Tool: Presentation 2 Demo"

    tkinter.messagebox.showinfo(title=title,
                                message="This is a demo of the standalone executable feature for the Spectrum "
                                        "Analyzer Tool created by Group 5.\n"
                                        "\nClick OK to select a video capture file to load...")

    csv_filepath: str = get_csv_filepath()
    launch_csv = tkinter.messagebox.askyesno(title=title,
                                             message=f"Processing Complete!\nCSV file saved at the "
                                                     f"following location: {csv_filepath}\n"
                                                     f"Would you like to open the file from the directory?")
    if launch_csv:
        os.system(f'start {csv_filepath}')


if __name__ == "__main__":
    # main()
    start()
