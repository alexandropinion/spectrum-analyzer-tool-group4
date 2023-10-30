
# Spectrum Analyzer Tool
___
This is a tool that can be utilized to process video captures of specific spectrum analyzers in order to generate csv data representing signals displayed on the analyzer's display.

## Description
Engineers and analysts at the 402 SWEG Robbins AFB are responsible for verifying frequency and amplitude data from a video of a Spectrum Analyzer representing an Aircraft Electronic Warfare (EW) System Under Test (SUT). The test could be a calibration or a test of the EW response to threat systems. The goal of the project is to reduce the manual toil involved in the review process and allow the transcribed data to be manipulated and further analyzed for anomalies. The current software project plan will provide a path for completing and managing the development and delivery of the provided software solution. This plan functions as a software development road map. It outlines all the procedures and information required to develop the program, including what it should do, who will work on it, when it will be completed, and how much it will cost. It keeps things organized and aids with everyone's understanding of what must be done, ensuring that the program is finished effectively and on schedule.

## Usage
- Open the executable
- Press the **Select Video** button to select a proper video files
- Press the **Load Video** button to begin processing
- Once processing is complete, a file location within a prompt should be displayed to show the resulting csv filepath

## Environment
If the customer would like to modify the existing code, the following enviroment can be used optionally to make revisions to source:

- Python 3.11_64bit (recommended to utilize a virtual environment)
- Install requirements.txt from the root folder in this project directory
- For UI changes, launch the designer with `qt5-tool designer` cmd from bash
  - Open any .ui file, make changes, save, and rebuild the associated .py file with:
    - `pyuic5 [ui_filename_here].ui -o [ui_filename_here].py` 
- Building an executable:
  - Once all changes are made, run the distribution.py file from this directory
    - Executable and all artifacts will be created into dist/[current_version_and_app_name_here].zip

# LICENSE
___
Copyright 2023 KSU

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

