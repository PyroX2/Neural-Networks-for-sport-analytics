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
import numpy as np
import torch


matplotlib.use("Qt5Agg")
envpath = '/home/jakub/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


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
        self.left_vbox.addWidget(self.image_window.pixmap_label)

        # VIDEO MANIPULATION LAYOUT
        self.video_layout = QtWidgets.QHBoxLayout()
        self.left_vbox.addLayout(self.video_layout)

        # Play Button
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setText("STOP")
        self.play_button.clicked.connect(self.change_play_button)
        self.play_button_status = True
        self.video_layout.addWidget(self.play_button)

        # SLIDER
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.video_layout.addWidget(self.slider)

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

        # KEYPOINT SELECTION
        self.selected_keypoint_label = QtWidgets.QLabel("Selected keypoints: ")
        self.selected_keypoint_combobox = QtWidgets.QComboBox()

        self.selected_keypoint_layout = QtWidgets.QHBoxLayout()
        self.selected_keypoint_layout.addWidget(
            self.selected_keypoint_label, alignment=QtCore.Qt.AlignRight)
        self.selected_keypoint_layout.addWidget(
            self.selected_keypoint_combobox, alignment=QtCore.Qt.AlignRight)

        for i in range(25):
            self.selected_keypoint_combobox.addItem(str(i))
        self.selected_keypoint_combobox.currentIndexChanged.connect(
            self._selected_keypoint_change)

        self.right_vbox.addLayout(
            self.selected_keypoint_layout)
        self.right_vbox.setAlignment(
            self.selected_keypoint_layout, QtCore.Qt.AlignRight)

        # VECTOR
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.vector = self.sc.axes.quiver(
            0, 0, 0, 0, angles='xy', scale_units='xy', scale=1)
        self.right_vbox.addWidget(self.sc)
        self.sc.axes.set_xlim(-1000, 1000)
        self.sc.axes.set_ylim(-1000, 1000)
        self.sc.axes.set_xlabel("px/s")
        self.sc.axes.set_ylabel("px/s")

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
        self.networks = {'YOLO': {"Enabled": 0, "Interface": yolo_interface.process_yolo, "Data": [], "Processed": False, "Keypoints": []},
                         'MediaPipe': {"Enabled": 0, "Interface": mediapipe_interface.process_mediapipe, "Data": [], "Processed": False, "Keypoints": []},
                         'OpenPifPaf': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": []},
                         'OpenPose': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": []}}
        self.network_index = 0
        self.file_path = ""
        self.processing = False
        self.selected_keypoint = 0
        self.prev_value = [0, 0]

    def selection_change(self, i):
        self.network_index = i
        # self.show_output()

    def check_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
            key = checkbox.text()
            self.networks[key]["Enabled"] = 1

    def checkbox_state(self):
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.networks[checkbox.text()]["Enabled"] = 1
            else:
                self.networks[checkbox.text()]["Enabled"] = 0

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.MainWindow, "File Dialog", "", "All Files (*);;Video Files (*.mp4);;Image Files (*.jpg)", options=options)
        if file_path:
            self.file_path = file_path

    def process_file(self):
        self.processing = True
        file_path, file_extension = os.path.split(self.file_path)
        if file_extension[-4:] == ".mp4":
            runtype = "Video"
        elif file_extension[-4:] == ".jpg":
            runtype = "Image"
        else:
            return

        for network in self.networks.keys():
            if self.networks[network]['Enabled'] == 0:
                self.networks[network]['Processed'] = True
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
        self.networks[output[2]]['Keypoints'] = output[1]
        self.networks[output[2]]['Processed'] = True
        self.progress_bar.setValue(100)
        for network in self.networks.keys():
            if not self.networks[network]['Processed']:
                return
        # This executes only if all networks are processed
        for network in self.networks.keys():
            self.networks[network]['Processed'] = False
        self.processing = False
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
        i = 0
        keypoint_position_prev = torch.tensor([0, 0])
        while True:
            start_time = time.time()
            self.slider.setValue(i)
            network = list(self.networks.keys())[self.network_index]
            data = self.networks[network]['Data']
            if self.networks[network]['Keypoints'][i].shape[0] > 0 and len(self.networks[network]['Keypoints'][i]) > self.selected_keypoint:
                keypoint_position = self.networks[network]['Keypoints'][i][self.selected_keypoint]
            else:
                keypoint_position = None
            number_of_frames = len(data)
            if len(self.networks[network]['Data']) > 0:
                frame = data[i]
            if keypoint_position is not None:
                processed_frame = cv2.circle(
                    copy.copy(frame), (int(keypoint_position[0]), int(keypoint_position[1])), 5, (255, 50, 50), 3)
                new_vector_value = self.calculate_new_vector(
                    keypoint_position, fps)
                self.move_vector(new_vector_value)
                self.prev_value = keypoint_position
                keypoint_position_prev = keypoint_position
            else:
                processed_frame = copy.copy(frame)
            self.image_window.set_image_data(
                cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            time.sleep(1/fps - (time.time() - start_time))

            # Calculate new vector based on the current and previous value
            i = self.slider.value()
            self.slider.setMaximum(number_of_frames-1)

            if self.play_button_status and i < len(self.networks[network]['Data'])-1:
                i += 1
            if self.processing:
                return

    def move_vector(self, new_value):
        self.vector.set_UVC(new_value[0], new_value[1])
        # self.sc.fig.canvas.draw()
        self.sc.axes.draw_artist(self.sc.axes.patch)
        self.sc.axes.draw_artist(self.vector)
        self.sc.fig.canvas.update()
        self.sc.fig.canvas.flush_events()

    def change_play_button(self):
        if self.play_button_status:
            self.play_button_status = False
            self.play_button.setText("PLAY")
        else:
            self.play_button_status = True
            self.play_button.setText("STOP")

    def calculate_new_vector(self, current_value, fps):
        new_vector_x = (current_value[0] - self.prev_value[0]) * fps
        new_vector_y = (current_value[1] - self.prev_value[1]) * fps
        return [new_vector_x, new_vector_y]

    def _selected_keypoint_change(self, i):
        self.selected_keypoint = int(i)

    def show(self):
        self.MainWindow.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()


if __name__ == '__main__':
    main()
