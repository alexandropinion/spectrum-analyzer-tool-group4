# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'signal_page.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1279, 836)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 0, 511, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 100, 491, 391))
        self.groupBox.setObjectName("groupBox")
        self.recordFreqDbm = QtWidgets.QCheckBox(self.groupBox)
        self.recordFreqDbm.setGeometry(QtCore.QRect(30, 20, 461, 61))
        self.recordFreqDbm.setObjectName("recordFreqDbm")
        self.recordMaxAmpl = QtWidgets.QCheckBox(self.groupBox)
        self.recordMaxAmpl.setGeometry(QtCore.QRect(30, 80, 321, 20))
        self.recordMaxAmpl.setObjectName("recordMaxAmpl")
        self.recordRelativeSignal = QtWidgets.QCheckBox(self.groupBox)
        self.recordRelativeSignal.setGeometry(QtCore.QRect(30, 120, 451, 20))
        self.recordRelativeSignal.setObjectName("recordRelativeSignal")
        self.recordAllOptions = QtWidgets.QCheckBox(self.groupBox)
        self.recordAllOptions.setGeometry(QtCore.QRect(30, 160, 281, 20))
        self.recordAllOptions.setObjectName("recordAllOptions")
        self.csvFilepathLabel = QtWidgets.QTextEdit(self.groupBox)
        self.csvFilepathLabel.setGeometry(QtCore.QRect(60, 300, 351, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.csvFilepathLabel.setFont(font)
        self.csvFilepathLabel.setReadOnly(True)
        self.csvFilepathLabel.setObjectName("csvFilepathLabel")
        self.csvFilepathBtn = QtWidgets.QPushButton(self.groupBox)
        self.csvFilepathBtn.setGeometry(QtCore.QRect(60, 240, 351, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.csvFilepathBtn.setFont(font)
        self.csvFilepathBtn.setObjectName("csvFilepathBtn")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 510, 491, 281))
        self.groupBox_2.setObjectName("groupBox_2")
        self.centerFreqID = QtWidgets.QTextEdit(self.groupBox_2)
        self.centerFreqID.setGeometry(QtCore.QRect(160, 40, 311, 31))
        self.centerFreqID.setObjectName("centerFreqID")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 131, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(20, 100, 131, 16))
        self.label_3.setObjectName("label_3")
        self.spanFreqID = QtWidgets.QTextEdit(self.groupBox_2)
        self.spanFreqID.setGeometry(QtCore.QRect(160, 90, 311, 31))
        self.spanFreqID.setObjectName("spanFreqID")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 131, 16))
        self.label_4.setObjectName("label_4")
        self.spanFreqID_2 = QtWidgets.QTextEdit(self.groupBox_2)
        self.spanFreqID_2.setGeometry(QtCore.QRect(160, 140, 311, 31))
        self.spanFreqID_2.setObjectName("spanFreqID_2")
        self.divisionSlider = QtWidgets.QSlider(self.groupBox_2)
        self.divisionSlider.setGeometry(QtCore.QRect(160, 220, 231, 22))
        self.divisionSlider.setMinimum(1)
        self.divisionSlider.setMaximum(30)
        self.divisionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.divisionSlider.setObjectName("divisionSlider")
        self.divisionLED = QtWidgets.QLCDNumber(self.groupBox_2)
        self.divisionLED.setGeometry(QtCore.QRect(410, 200, 61, 61))
        self.divisionLED.setObjectName("divisionLED")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(10, 210, 131, 41))
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_9")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(580, 110, 471, 381))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.signalImage = QtWidgets.QLabel(self.groupBox_3)
        self.signalImage.setGeometry(QtCore.QRect(70, 30, 671, 291))
        self.signalImage.setText("")
        self.signalImage.setObjectName("signalImage")
        self.signalSlider = QtWidgets.QSlider(self.centralwidget)
        self.signalSlider.setGeometry(QtCore.QRect(580, 70, 471, 22))
        self.signalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.signalSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.signalSlider.setObjectName("signalSlider")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(580, 10, 261, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.centerFreq = QtWidgets.QLCDNumber(self.centralwidget)
        self.centerFreq.setGeometry(QtCore.QRect(1170, 120, 81, 71))
        self.centerFreq.setObjectName("centerFreq")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1060, 150, 101, 20))
        self.label_6.setObjectName("label_6")
        self.spanFreq = QtWidgets.QLCDNumber(self.centralwidget)
        self.spanFreq.setGeometry(QtCore.QRect(1170, 210, 81, 71))
        self.spanFreq.setObjectName("spanFreq")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1060, 240, 101, 20))
        self.label_7.setObjectName("label_7")
        self.refLevel = QtWidgets.QLCDNumber(self.centralwidget)
        self.refLevel.setGeometry(QtCore.QRect(1170, 310, 81, 71))
        self.refLevel.setObjectName("refLevel")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(1060, 330, 101, 20))
        self.label_8.setObjectName("label_8")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(580, 510, 671, 281))
        self.groupBox_4.setObjectName("groupBox_4")
        self.goBackBtn = QtWidgets.QPushButton(self.groupBox_4)
        self.goBackBtn.setGeometry(QtCore.QRect(20, 40, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.goBackBtn.setFont(font)
        self.goBackBtn.setObjectName("goBackBtn")
        self.startProcessingBtn = QtWidgets.QPushButton(self.groupBox_4)
        self.startProcessingBtn.setGeometry(QtCore.QRect(20, 120, 321, 111))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.startProcessingBtn.setFont(font)
        self.startProcessingBtn.setObjectName("startProcessingBtn")
        self.dbPerDivLED = QtWidgets.QLCDNumber(self.centralwidget)
        self.dbPerDivLED.setGeometry(QtCore.QRect(1170, 410, 81, 71))
        self.dbPerDivLED.setObjectName("dbPerDivLED")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1060, 430, 101, 20))
        self.label_10.setObjectName("label_10")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Select Signal Process Decisions:"))
        self.groupBox.setTitle(_translate("MainWindow", "Select CSV Recording Options:"))
        self.recordFreqDbm.setText(_translate("MainWindow", "Record entire signal with scaling factors  (frequency and ampltiude applied)"))
        self.recordMaxAmpl.setText(_translate("MainWindow", "Record only max amplitude with scaling factors"))
        self.recordRelativeSignal.setText(_translate("MainWindow", "Record entire signal with no scaling factors (relative coordinate values)"))
        self.recordAllOptions.setText(_translate("MainWindow", "Record all options"))
        self.csvFilepathBtn.setText(_translate("MainWindow", "Press to Change CSV Filepath"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Select Signal IDs (names used for signal characteristics on Spectrum Analyzer):"))
        self.label_2.setText(_translate("MainWindow", "Center Frequency ID:"))
        self.label_3.setText(_translate("MainWindow", "Span Frequency ID:"))
        self.label_4.setText(_translate("MainWindow", "Reference Level ID:"))
        self.label_9.setText(_translate("MainWindow", "Total Signal (y axis) Divisions:"))
        self.label_5.setText(_translate("MainWindow", "Select Frame:"))
        self.label_6.setText(_translate("MainWindow", "Center Freq (ghz):"))
        self.label_7.setText(_translate("MainWindow", "Span Freq (mhz):"))
        self.label_8.setText(_translate("MainWindow", "Ref Level (dbm):"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Controls"))
        self.goBackBtn.setText(_translate("MainWindow", "< Go back to Cal: Template"))
        self.startProcessingBtn.setText(_translate("MainWindow", "Start Processing"))
        self.label_10.setText(_translate("MainWindow", "dB per Division:"))
