import PyQt5
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import os
import mediapipe_interface
import yolo_interface
import cv2
from worker import Worker
from image_window import ImageWindow


matplotlib.use("Qt5Agg")
envpath = '/home/jakub/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.initialize_variables()

        # MAIN PAGE
        self.MainWindow = QtWidgets.QWidget()
        self.MainWindow.resize(1000, 600)
        self.MainWindow.setStyleSheet("background-color: white;")
        self.MainWindow.setWindowTitle("NNs for sport analytics")
        self.main_page_layout = QtWidgets.QGridLayout()

        # LAYOUTS
        self.left_vbox = QtWidgets.QVBoxLayout()
        self.right_vbox = QtWidgets.QVBoxLayout()

        self.main_page_layout.addLayout(self.left_vbox, 0, 0)
        self.main_page_layout.addLayout(self.right_vbox, 0, 1)

        # CHECKBOX
        self.checkbox_layout = QtWidgets.QHBoxLayout()

        self.checkboxes = []

        self.check_all_button = QtWidgets.QPushButton()
        self.check_all_button.setText("Check All")
        self.check_all_button.clicked.connect(self.check_all)
        self.checkbox_layout.addWidget(self.check_all_button)

        for i, key in enumerate(self.networks.keys()):
            checkbox = QtWidgets.QCheckBox(key)
            checkbox.setChecked(False)
            self.checkboxes.append(checkbox)
            checkbox.stateChanged.connect(self.checkbox_state)
            self.networks[key]["Enabled"] = 0
            self.checkbox_layout.addWidget(checkbox)

        self.left_vbox.addLayout(
            self.checkbox_layout)

        # CURRENTLY DISPLAYED NETWORK
        self.currently_displayed = QtWidgets.QLabel("Currently displayed: ")
        self.currently_displayed_combobox = QtWidgets.QComboBox()

        self.currently_displayed_layout = QtWidgets.QHBoxLayout()
        self.currently_displayed_layout.addWidget(
            self.currently_displayed, alignment=QtCore.Qt.AlignRight)
        self.currently_displayed_layout.addWidget(
            self.currently_displayed_combobox, alignment=QtCore.Qt.AlignRight)

        for key in self.networks.keys():
            self.currently_displayed_combobox.addItem(key)
        self.currently_displayed_combobox.currentIndexChanged.connect(
            self.selection_change)

        self.right_vbox.addLayout(
            self.currently_displayed_layout)
        self.right_vbox.setAlignment(
            self.currently_displayed_layout, QtCore.Qt.AlignRight)

        # IMAGE WINDOW
        self.image_window = ImageWindow()
        # self.current_network = QtWidgets.QLabel()
        # self.current_network.setText(
        # list(self.networks.keys())[self.network_index])
        self.left_vbox.addWidget(self.image_window.pixmap_label)

        # FILE DIALOG BUTTON
        self.file_dialog_button = QtWidgets.QPushButton()
        self.file_dialog_button.clicked.connect(self.openFileNameDialog)
        self.file_dialog_button.setText("Choose file")
        self.right_vbox.addWidget(self.file_dialog_button)

        # PROCESS FILE BUTTON
        self.process_button = QtWidgets.QPushButton()
        self.process_button.clicked.connect(self.process_file)
        self.process_button.setText("Process")
        self.process_button.setStyleSheet("background-color : #08c71b")
        self.right_vbox.addWidget(self.process_button)

        # PROGRESS BAR
        self.progress_bar = QtWidgets.QProgressBar(self.MainWindow)
        self.progress_bar.setGeometry(30, 40, 200, 25)
        self.right_vbox.addWidget(self.progress_bar)

        # LOGO
        self.logo = QtWidgets.QLabel()
        self.pixmap = QtGui.QPixmap(
            '/home/jakub/inzynierka/app/images/Qt_logo.png')
        self.pixmap = self.pixmap.scaled(100, 75)
        self.logo.setPixmap(self.pixmap)

        self.right_vbox.addWidget(
            self.logo)
        self.right_vbox.setAlignment(
            self.logo, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

        # MAIN PAGE LAYOUT
        self.MainWindow.setLayout(self.main_page_layout)

        # MULTITHREADING
        self.threadpool = QtCore.QThreadPool()

    def initialize_variables(self):
        self.networks = {'YOLO': {"Enabled": 0, "Interface": yolo_interface.process_yolo, "Data": [], "Processed": False},
                         'MediaPipe': {"Enabled": 0, "Interface": mediapipe_interface.process_mediapipe, "Data": [], "Processed": False},
                         'OpenPifPaf': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False},
                         'OpenPose': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False}}
        self.network_index = 0
        self.file_path = ""

    def selection_change(self, i):
        self.network_index = i
        self.show_output()

    def check_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
            key = checkbox.text()
            self.networks[key]["Enabled"] = 1
            print(self.networks)

    def checkbox_state(self):
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.networks[checkbox.text()]["Enabled"] = 1
            else:
                self.networks[checkbox.text()]["Enabled"] = 0
        print(self.networks)

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.MainWindow, "File Dialog", "", "All Files (*);;Video Files (*.mp4);;Image Files (*.jpg)", options=options)
        if file_path:
            self.file_path = file_path
            print(self.file_path)

    def process_file(self):
        file_path, file_extension = os.path.split(self.file_path)
        if file_extension[-4:] == ".mp4":
            runtype = "Video"
        elif file_extension[-4:] == ".jpg":
            runtype = "Image"
        else:
            print(file_extension)
            return

        for network in self.networks.keys():
            if self.networks[network]['Enabled'] == 0:
                self.networks[network]['Data'] = []
                continue
            network_interface = self.networks[network]['Interface']
            if network_interface is None:
                continue
            worker = Worker(network_interface,
                            runtype, self.file_path, self.progress_bar)
            self.threadpool.start(worker)
            worker.signals.result.connect(self.save_output)

    def save_output(self, output):
        self.networks[output[2]]['Data'] = output[0]
        self.networks[output[2]]['Processed'] = True
        self.progress_bar.setValue(100)
        self.show_output()

    def show_output(self):
        network = list(self.networks.keys())[self.network_index]
        data = self.networks[network]['Data']
        if len(data) == 1:
            self.show_image()
        elif len(data) > 1:
            worker = Worker(self.show_video)
            self.threadpool.start(worker)
        else:
            self.progress_bar.reset()
            self.image_window.reset()
            return
        self.progress_bar.reset()

    def show_image(self):
        network = list(self.networks.keys())[self.network_index]
        data = self.networks[network]['Data']
        self.image_window.set_image_data(
            cv2.cvtColor(data[0], cv2.COLOR_RGB2BGR))

    def show_video(self):
        video = cv2.VideoCapture(self.file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        network = list(self.networks.keys())[self.network_index]
        data = self.networks[network]['Data']
        for frame in data:
            self.image_window.set_image_data(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            time.sleep(1/fps)
            if list(self.networks.keys())[self.network_index] != network:
                break
        # QtGui.QPainter().end()

    def show(self):
        self.MainWindow.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()


if __name__ == '__main__':
    main()
