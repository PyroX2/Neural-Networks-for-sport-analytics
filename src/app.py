from PyQt6 import QtWidgets, QtCore
import sys
import copy
import time
import matplotlib
import os
import mediapipe_interface
import yolo_interface
import cv2
from worker import Worker
from image_window import ImageWindow
import numpy as np
import torch
from plt_canvas import PltCanvas

matplotlib.use("Qt5Agg")
user_name = os.getenv("USER")
envpath = f'/home/{user_name}/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()

        # Initialize variables besides GUI widgets
        self.initialize_variables()

        # MAIN PAGE
        self.central_widget = QtWidgets.QWidget()
        self.central_widget.resize(1000, 600)
        self.central_widget.setWindowTitle("NNs for sport analytics")

        # Initialize layouts
        self.main_page_layout = QtWidgets.QGridLayout()  # Grid layout for main page
        self.left_vbox = QtWidgets.QVBoxLayout()  # Left vertical box
        self.right_vbox = QtWidgets.QVBoxLayout()  # Right vertical box
        # Add left vbox to main layout
        self.main_page_layout.addLayout(self.left_vbox, 0, 0)
        # Add right vbox to main layout
        self.main_page_layout.addLayout(self.right_vbox, 0, 1)

        # NEURAL NETWORK CHECKBOX
        self.checkbox_layout = QtWidgets.QHBoxLayout()

        # Button to check all neural networks as active
        check_all_button = QtWidgets.QPushButton()
        check_all_button.setText("Check All")
        check_all_button.clicked.connect(self.check_all_networks)
        self.checkbox_layout.addWidget(check_all_button)

        self.checkboxes = []

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
            self.currently_displayed, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.currently_displayed_layout.addWidget(
            self.currently_displayed_combobox, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        for key in self.networks.keys():
            self.currently_displayed_combobox.addItem(key)
        self.currently_displayed_combobox.currentIndexChanged.connect(
            self.network_selection_change)

        self.right_vbox.addLayout(
            self.currently_displayed_layout)
        self.right_vbox.setAlignment(
            self.currently_displayed_layout, QtCore.Qt.AlignmentFlag.AlignRight)

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
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.video_layout.addWidget(self.slider)

        # PREV FRAME
        self.prev_frame = QtWidgets.QPushButton()
        self.prev_frame.setText("-")
        self.prev_frame.clicked.connect(self._prev_frame)
        self.video_layout.addWidget(self.prev_frame)

        # NEXT FRAME
        self.next_frame = QtWidgets.QPushButton()
        self.next_frame.setText("+")
        self.next_frame.clicked.connect(self._next_frame)
        self.video_layout.addWidget(self.next_frame)

        # FILE DIALOG BUTTON
        self.file_dialog_button = QtWidgets.QPushButton()
        self.file_dialog_button.clicked.connect(self.open_file_dialog)
        self.file_dialog_button.setText("Choose file")
        self.right_vbox.addWidget(self.file_dialog_button)

        # PROCESS FILE BUTTON
        self.process_button = QtWidgets.QPushButton()
        self.process_button.clicked.connect(self.process_file)
        self.process_button.setText("Process")
        self.process_button.setStyleSheet("background-color : #08c71b")
        self.right_vbox.addWidget(self.process_button)

        # PROGRESS BAR
        self.progress_bar = QtWidgets.QProgressBar(self.central_widget)
        self.progress_bar.setGeometry(30, 40, 200, 25)
        self.right_vbox.addWidget(self.progress_bar)

        # PLOT SELECTION
        self.selected_plot_label = QtWidgets.QLabel("Selected plot: ")
        self.selected_plot_combobox = QtWidgets.QComboBox()
        self.selected_plot_combobox.addItem("Velocity")
        self.selected_plot_combobox.addItem("3D Plot")
        self.selected_plot_combobox.currentIndexChanged.connect(
            self._selected_plot_change)

        # KEYPOINT SELECTION
        self.selected_keypoint_label = QtWidgets.QLabel("Selected keypoints: ")
        self.selected_keypoint_combobox = QtWidgets.QComboBox()

        for i in range(25):
            self.selected_keypoint_combobox.addItem(str(i))
        self.selected_keypoint_combobox.currentIndexChanged.connect(
            self._selected_keypoint_change)

        self.selected_keypoint_layout = QtWidgets.QHBoxLayout()
        # Add widgets
        self.selected_keypoint_layout.addWidget(
            self.selected_plot_label, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.selected_keypoint_layout.addWidget(
            self.selected_plot_combobox, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.selected_keypoint_layout.addWidget(
            self.selected_keypoint_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.selected_keypoint_layout.addWidget(
            self.selected_keypoint_combobox, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        self.right_vbox.addLayout(
            self.selected_keypoint_layout)
        self.right_vbox.setAlignment(
            self.selected_keypoint_layout, QtCore.Qt.AlignmentFlag.AlignRight)

        # VECTOR
        self.sc = PltCanvas(width=5, height=4, dpi=100)
        self.vector = self.sc.axes.quiver(
            0, 0, 0, 0, angles='xy', scale_units='xy', scale=1)
        self.right_vbox.addWidget(self.sc)

        # 2D PLOT
        self._2d_plot = self.sc.axes.scatter([0], [0])

        # MAIN PAGE LAYOUT
        self.central_widget.setLayout(self.main_page_layout)

        # MULTITHREADING
        self.threadpool = QtCore.QThreadPool()

        self.setCentralWidget(self.central_widget)

        self.show()

    # Initialize variables not related to GUI
    def initialize_variables(self):
        # Dictionary for data related to each neural network
        self.networks = {'YOLO': {"Enabled": 0, "Interface": yolo_interface.process_yolo, "Data": [], "Processed": False, "Keypoints": []},
                         'MediaPipe': {"Enabled": 0, "Interface": mediapipe_interface.process_mediapipe, "Data": [], "Processed": False, "Keypoints": []},
                         'OpenPifPaf': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": []},
                         'OpenPose': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": []}}
        self.network_index = 0  # Currently selected network to be displayed
        self.file_path = ""  # File path to be processed
        self.processing = False  # Is the image currently being processed by the NNs
        self.selected_keypoint = 0  # Currently selected keypoint to be displayed on some plots
        self.selected_plot = 0  # Currently dipslayed MatPlotLib plot

        # Previous position value for caluclating velocity
        self.pos_prev_value = [0, 0]

        # Currently displayed frame
        self.current_frame = 0

    # Change network index to be displayed
    def network_selection_change(self, i):
        self.network_index = i  # This index is selecting the network by the self.networks.keys()

    # Function to check all neural networks as active to be processed
    def check_all_networks(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
            key = checkbox.text()
            self.networks[key]["Enabled"] = 1

    # Read if neural network is set as active to be processed and write it to self.networks dictionary
    def checkbox_state(self):
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.networks[checkbox.text()]["Enabled"] = 1
            else:
                self.networks[checkbox.text()]["Enabled"] = 0

    # Function to open file dialog and select file to be processed
    def open_file_dialog(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setDirectory('./')  # Open dialog with current directory
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        # Set view as a list
        dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()

            # Set file path as first selected file
            self.file_path = filenames[0]

    # Function for starting workers with selected file and NNs
    def process_file(self):
        self.processing = True  # Set processing flag as True
        file_path, file_extension = os.path.split(
            self.file_path)  # Extract file extension

        # Set correct runtype
        if file_extension[-4:] == ".mp4":
            runtype = "Video"
        elif file_extension[-4:] == ".jpg":
            runtype = "Image"
        else:
            return '''TODO: ERROR SHOULD BE DISPLAYED'''

        for network in self.networks.keys():
            # If network is not enabled continue to next one
            if self.networks[network]['Enabled'] == 0:
                self.networks[network]['Processed'] = True
                self.networks[network]['Data'] = []
                continue

            # Use interface for selected networks
            network_interface = self.networks[network]['Interface']

            # Check if interface is provided
            if network_interface is None:
                raise RuntimeError(
                    "Selected network doesn't have a proper interface for processing")

            # Start the worker with proper arguments
            worker = Worker(network_interface,
                            runtype, self.file_path)
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
        self.current_frame = 0
        while True:
            start_time = time.time()
            self.slider.setValue(self.current_frame)
            network = list(self.networks.keys())[self.network_index]
            data = self.networks[network]['Data']
            print(f'FRAME SHAPE: {np.array(self.networks[network]
                  ['Keypoints'][self.current_frame]).shape}')
            if self.networks[network]['Keypoints'][self.current_frame].shape[0] > 0 and len(self.networks[network]['Keypoints'][self.current_frame]) > self.selected_keypoint:
                keypoint_position = self.networks[network]['Keypoints'][self.current_frame][self.selected_keypoint]
            else:
                keypoint_position = None
            number_of_frames = len(data)
            if len(self.networks[network]['Data']) > 0:
                frame = data[self.current_frame]

            if keypoint_position is not None:
                processed_frame = cv2.circle(
                    copy.copy(frame), (int(keypoint_position[0]), int(keypoint_position[1])), 5, (255, 50, 50), 3)
            else:
                processed_frame = frame

            self.sc.axes.draw_artist(self.sc.axes.patch)
            if self.selected_plot == 0:
                self.sc.axes.set_xlim(-1000, 1000)
                self.sc.axes.set_ylim(-1000, 1000)
                self.sc.axes.set_xlabel("px/s")
                self.sc.axes.set_ylabel("px/s")
                self._process_vector(
                    frame, keypoint_position, fps)
                self.sc.axes.draw_artist(self.vector)
            elif self.selected_plot == 1:
                self.sc.axes.set_xlim(0, frame.shape[0])
                self.sc.axes.set_ylim(0, frame.shape[0])
                keypoints = self.networks[network]['Keypoints'][self.current_frame]
                self._process_3d(frame.shape[1], keypoints)
                self.sc.axes.draw_artist(self._2d_plot)
            self.sc.fig.canvas.update()
            self.sc.fig.canvas.flush_events()

            self.image_window.set_image_data(
                cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
            sleep_time = 1/fps - (time.time() - start_time)
            if sleep_time >= 0:
                time.sleep(sleep_time)

            self.current_frame = self.slider.value()
            self.slider.setMaximum(number_of_frames-1)

            if self.play_button_status and self.current_frame < len(self.networks[network]['Data'])-1:
                self.current_frame += 1
            if self.processing:
                return

    def move_vector(self, new_value):
        self.vector.set_UVC(new_value[0], new_value[1])
        # self.sc.fig.canvas.draw()

    def change_play_button(self):
        if self.play_button_status:
            self.play_button_status = False
            self.play_button.setText("PLAY")
        else:
            self.play_button_status = True
            self.play_button.setText("STOP")

    def calculate_new_vector(self, current_value, fps):
        new_vector_x = (current_value[0] - self.pos_prev_value[0]) * fps
        new_vector_y = (current_value[1] - self.pos_prev_value[1]) * fps
        return [new_vector_x, new_vector_y]

    def _selected_keypoint_change(self, i):
        self.selected_keypoint = int(i)

    def _selected_plot_change(self, i):
        self.selected_plot = int(i)

    def _process_vector(self, frame, keypoint_position, fps):
        self.sc.axes.set_xlim(-1000, 1000)
        self.sc.axes.set_ylim(-1000, 1000)
        self.sc.axes.set_xlabel("px/s")
        self.sc.axes.set_ylabel("px/s")
        if keypoint_position is not None:

            new_vector_value = self.calculate_new_vector(
                keypoint_position, fps)
            self.move_vector(new_vector_value)
            self.pos_prev_value = keypoint_position

    def _process_3d(self, frame_y_shape, keypoints):
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        y = torch.tensor([frame_y_shape]*len(y)) - y + 200
        self._2d_plot.set_offsets(np.c_[x, y])

    def _prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.setValue(self.current_frame)

    def _next_frame(self):
        network = list(self.networks.keys())[self.network_index]
        if self.current_frame < len(self.networks[network]['Data'])-1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    app.exec()


if __name__ == '__main__':
    main()
