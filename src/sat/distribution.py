#!/usr/bin/env python3

#: Imports
import logging
import sys
import os
import PyInstaller.__main__
import shutil

#: Metadata
__app_version__ = '0.0.2'
__app_name__ = 'Spectrum Analyzer Tool'

#: Globals
TEMP_EXE_DIRNAME = "data"  # This is the folder name that is stored in the "temp" folder when an executable version
# of this application is run on a Windows OS.
README_FILENAME = "README.md"


#: Functions
def build_distribution() -> None:
    #: (1) Get OS type
    os_type = sys.platform
    os_separator: str = ""
    if os_type == 'win32':
        os_separator = ";"
    else:  # macOS/Linux distros utilize a different separator for pyinstaller
        os_separator = ":"

    #: (2) Grab directories
    curr_path = os.getcwd()
    curr_parent_path = os.path.abspath(os.path.join(curr_path, os.pardir))
    logging.info(f"Current path: {curr_path}\nCurrent path's parent: {curr_parent_path}")

    #: (3) Build the standalone executable/app
    dist_fp = f"{curr_parent_path}/dist/"
    dist_name = f"{__app_name__} v{__app_version__}"
    exe_fp = f"{dist_fp}/{dist_name}"
    PyInstaller.__main__.run([
        '--name=%s' % __app_name__,
        '--onefile',
        '--noconsole',
        f'--workpath=%s' % f"{dist_fp}",
        f'--distpath=%s' % f"{exe_fp}",
        '--add-data=%s' % f"*.py{os_separator}.",
        '--add-data=%s' % f"*.ui{os_separator}.",
        os.path.join('', 'run.py'),
    ])

    #: Create the zip we want to distribute and save to a desired location.
    shutil.copy(f"{curr_path}/{README_FILENAME}", f"{exe_fp}")
    shutil.copy(f"{curr_path}/imgs/template_1.png", f"{exe_fp}")
    shutil.copy(f"{curr_path}/config.ini", f"{exe_fp}")
    shutil.make_archive(dist_name, 'zip', exe_fp)
    shutil.copy(f"{curr_path}/{dist_name}.zip", dist_fp)

    #: Delete garbage build artifacts
    files_in_curr_dir = os.listdir(curr_path)
    for item in files_in_curr_dir:
        if item.endswith(".spec"):
            os.remove(os.path.join(curr_path, item))
        if item.endswith(".zip"):
            os.remove(os.path.join(curr_path, item))
    shutil.rmtree(f"{dist_fp}{dist_name}/",
                  ignore_errors=True)  # Remove the "non" zip distribution folder.
    shutil.rmtree(f"{dist_fp}/{__app_name__}/",
                  ignore_errors=True)  # Remove the working folder used to build the executable.


#: Main entry
if __name__ == "__main__":
    build_distribution()
