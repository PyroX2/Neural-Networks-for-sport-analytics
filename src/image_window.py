import PyQt5
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage
import cv2
import sys
import os
import numpy as np


envpath = '/home/jakub/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class ImageWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        # self.nn_image = QImage(
        #     image, image.shape[0], image.shape[1], image.shape[0]*3, QImage.Format_BGR888)
        self.pixmap_label = QtWidgets.QLabel()
        # self.nn_pixmap = QtGui.QPixmap(self.nn_image)
        # self.pixmap_label.setPixmap(self.nn_pixmap)
        # self.main_page_layout.addWidget(self.pixmap_label)

        # self.main_widget.setLayout(self.main_page_layout)

    def set_image_data(self, data):
        image = np.array(data)
        image = image.reshape(image.shape[1], image.shape[0], image.shape[2])
        nn_image = QImage(
            image, image.shape[0], image.shape[1], image.shape[0]*3, QImage.Format_BGR888)
        nn_pixmap = QtGui.QPixmap(nn_image)
        self.pixmap_label.setPixmap(nn_pixmap)

    def reset(self):
        pixmap = QtGui.QPixmap()
        self.pixmap_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = ImageWindow()
    gui.show()
    app.exec_()
