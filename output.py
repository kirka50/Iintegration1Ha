# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg


class Ui_MainWindow(object):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        self.CentralWidg = QtWidgets.QWidget(MainWindow)
        self.CentralWidg.setObjectName("CentralWidg")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.CentralWidg)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.DownloadButton = QtWidgets.QPushButton(self.CentralWidg)
        self.DownloadButton.setObjectName("DownloadButton")
        self.verticalLayout_2.addWidget(self.DownloadButton)

        self.DownloadButton.setStyleSheet("background-color : lightgrey")
        self.CheckButton = QtWidgets.QPushButton(self.CentralWidg)
        self.CheckButton.setObjectName("CheckButton")
        self.verticalLayout_2.addWidget(self.CheckButton)
        self.CheckButton.setStyleSheet("background-color : lightgrey")
        self.CheckButton.clicked.connect(self.changeColor)
        self.CheckButton.setCheckable(True)
        self.scrollArea = QtWidgets.QScrollArea(self.CentralWidg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 500))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.DownloadButton.clicked.connect(self.openFileNamesDialog)
        self.scrollArea.setObjectName("scrollArea")
        self.ScrollAreaContent = QtWidgets.QWidget()
        self.ScrollAreaContent.setGeometry(QtCore.QRect(0, 0, 1243, 1000))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ScrollAreaContent.sizePolicy().hasHeightForWidth())
        self.ScrollAreaContent.setSizePolicy(sizePolicy)
        self.ScrollAreaContent.setMinimumSize(QtCore.QSize(0, 1000))
        self.ScrollAreaContent.setObjectName("ScrollAreaContent")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.ScrollAreaContent)
        self.verticalLayout.setObjectName("verticalLayout")
        '''
        self.test = QtWidgets.QLabel(self.ScrollAreaContent)
        self.test.setObjectName("test")
        self.verticalLayout.addWidget(self.test)

        
        self.test1 = QtWidgets.QLabel(self.ScrollAreaContent)
        self.test1.setObjectName("test")
        self.verticalLayout.addWidget(self.test1)
        '''

        self.scrollArea.setWidget(self.ScrollAreaContent)
        self.verticalLayout_2.addWidget(self.scrollArea)
        self.PlotWidget = pg.PlotWidget()
        self.PlotWidget.setBackground('Black')
        self.PlotWidget.plot(self.x, self.y)
        self.verticalLayout.addWidget(self.PlotWidget)
        self.PlotWidget1 = pg.PlotWidget()
        self.PlotWidget1.setBackground('Black')
        self.PlotWidget1.plot(self.x, self.y)
        self.verticalLayout.addWidget(self.PlotWidget1)
        self.PlotWidget2 = pg.PlotWidget()
        self.PlotWidget2.setBackground('Black')
        self.PlotWidget2.plot(self.x, self.y)
        self.verticalLayout.addWidget(self.PlotWidget2)

        MainWindow.setCentralWidget(self.CentralWidg)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.DownloadButton.setText(_translate("MainWindow", "?????????????????? ????????????"))
        self.CheckButton.setText(_translate("MainWindow",'?????????? ???????????????????? ????????????????'))
        'self.test.setText(_translate("MainWindow", "TextLabel"))'


    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(MainWindow, "???????????????? ?????????????????? ?????? ???????? Exel ????????", "",
                                                "Exel files(*.xlsx)", options=options)
        if files:
            print(files)

    def changeColor(self):

        # if button is checked
        if self.CheckButton.isChecked():

            # setting background color to light-blue
            self.CheckButton.setStyleSheet("background-color : lightgreen")

        # if it is unchecked
        else:

            # set background color back to light-grey
            self.CheckButton.setStyleSheet("background-color : lightgrey")

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
