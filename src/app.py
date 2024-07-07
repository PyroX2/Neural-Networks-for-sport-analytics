from PyQt6 import QtWidgets, QtCore, QtGui
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
from video_worker import VideoWorker
from settings import Settings
import utils


matplotlib.use("Qt5Agg")
user_name = os.getenv("USER")
envpath = f'/home/{user_name}/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

TOTAL_NUMBER_OF_KEYPOINTS = 25


class GUI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(QtWidgets.QMainWindow, self).__init__()

        # Initialize variables besides GUI widgets
        self.initialize_variables()

        # MAIN PAGE
        self.central_widget = QtWidgets.QWidget()
        self.resize(5000, 5000)
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
        self.left_vbox.addWidget(
            self.image_window.pixmap_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

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
        self.slider.valueChanged.connect(self._set_current_frame)
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
        self.selected_plot_combobox.addItem("2D Plot")
        self.selected_plot_combobox.currentIndexChanged.connect(
            self._selected_plot_change)

        # KEYPOINT SELECTION
        self.selected_keypoint_label = QtWidgets.QLabel("Selected keypoints: ")
        self.selected_keypoint_combobox = QtWidgets.QComboBox()

        for i in range(TOTAL_NUMBER_OF_KEYPOINTS):
            self.selected_keypoint_combobox.addItem(str(i))
        self.selected_keypoint_combobox.currentIndexChanged.connect(
            self._selected_keypoint_change)
        self.selected_keypoint_combobox.setStyleSheet("QComboBox"
                                                      "{"
                                                      "background-color: darkblue;"
                                                      "}")
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
    def initialize_variables(self) -> None:
        # Dictionary for data related to each neural network
        self.networks = {'YOLO': {"Enabled": 0, "Interface": yolo_interface.process_yolo, "Data": [], "Processed": False, "Keypoints": [], 'Progress_bar_value': 0},
                         'MediaPipe': {"Enabled": 0, "Interface": mediapipe_interface.process_mediapipe, "Data": [], "Processed": False, "Keypoints": [], 'Progress_bar_value': 0},
                         'OpenPifPaf': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": [], 'Progress_bar_value': 0},
                         'OpenPose': {"Enabled": 0, "Interface": None, "Data": [], "Processed": False, "Keypoints": [], 'Progress_bar_value': 0}}
        self.network_index = 0  # Currently selected network to be displayed
        self.file_path = ""  # File path to be processed
        self.processing = False  # Is the image currently being processed by the NNs
        self.selected_keypoint = 0  # Currently selected keypoint to be displayed on some plots
        self.selected_plot = 0  # Currently dipslayed MatPlotLib plot

        # Previous position value for caluclating velocity
        self.pos_prev_value = torch.zeros((3, 1))

        # Currently displayed frame
        self.current_frame = 0

        # Add settings
        self.settings = Settings()

    # Change network index to be displayed
    def network_selection_change(self, i: int) -> None:
        self.network_index = i  # This index is selecting the network by the self.networks.keys()

        if self.current_frame == 0:
            # Gets data for currently selected neural network
            data = self._get_network_data()
            self.number_of_frames = len(data)
            self.show_output(data)

    # Function to check all neural networks as active to be processed
    def check_all_networks(self) -> None:
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
            key = checkbox.text()
            self.networks[key]["Enabled"] = 1

    # Read if neural network is set as active to be processed and write it to self.networks dictionary
    def checkbox_state(self) -> None:
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.networks[checkbox.text()]["Enabled"] = 1
            else:
                self.networks[checkbox.text()]["Enabled"] = 0

    # Function to open file dialog and select file to be processed
    def open_file_dialog(self) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setDirectory('./')  # Open dialog with current directory
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        # Set view as a list
        dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()

            # Set file path as first selected file
            self.file_path = filenames[0]

    # Function for starting workers with selected file and NNs activated in checkboxes
    def process_file(self) -> None:
        self.progress_bar.reset()  # Reset progress bar
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
            worker.signals.progress_bar.connect(self._update_progress_bar)

    # Function activated when workers responsible for passing videos through neural networks finish their work.
    # This function saves outputs in appropriate places in self.networks dictionary
    def save_output(self, output: tuple) -> None:
        # Saves processed frames with skeletons drawn
        self.networks[output[2]]['Data'] = output[0]  # List

        # Saves keypoints positions on each frame
        self.networks[output[2]]['Keypoints'] = output[1]  # List

        # Indicates that processing of this neural network is finished
        self.networks[output[2]]['Processed'] = True

        # Checks if all neural networks finished processing. If not returns
        for network in self.networks.keys():
            if not self.networks[network]['Processed']:
                return

        # If all neural networks finished processing set progress bar value to 100
        self.progress_bar.setValue(100)

        # Reset 'Processed' flags for all NNs
        for network in self.networks.keys():
            self.networks[network]['Processed'] = False

        self.processing = False  # Flag indicating that processing is finished

        self.current_frame = 0  # Resets the video

        # Gets data for currently selected neural network
        data = self._get_network_data()

        # Reads number of total frames for slider
        self.number_of_frames = len(data)

        # Show output
        self.show_output(data)

    # Show to output by calling appropriate function
    def show_output(self, data: list) -> None:
        if len(data) == 1:
            self._show_image(data)
        elif len(data) > 1:
            # Starts worker that emits processed frames
            worker = VideoWorker(self.process_frame)

            # Connection to receive processed frames
            worker.signals.result.connect(self._show_video_output)

            self.threadpool.start(worker)  # Starts worker
        else:
            self.image_window.reset()

    # Displays image
    def _show_image(self, data: list) -> None:
        frame = utils.scale_frame(
            data[0], self.settings.x_max_image_size, self.settings.y_max_image_size)

        # Display the image
        self.image_window.set_image_data(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Function responsible for processing video frames by the worker
    def process_frame(self) -> tuple[np.array, bool]:
        video = cv2.VideoCapture(self.file_path)  # Reads video
        fps = video.get(cv2.CAP_PROP_FPS)  # Reads video's fps

        # Reads which neural network to use
        network = self._get_network_name()

        # Reads current neural network data
        data = self._get_network_data()

        # Reads current neural network keypoints
        keypoints = self._get_network_keypoints()

        # Reads current frame if frames for current neural network exist
        if len(data) > 0:
            frame = data[self.current_frame]

            # Checks if keypoints for the current frame exist and number of keypoints is larger than selected for display keypoint index
            if self._get_network_keypoints()[self.current_frame].shape[0] > 0 and len(self._get_network_keypoints()[self.current_frame]) > self.selected_keypoint:
                # Only one keypoint position (currently selected one)
                keypoint_position = self._get_network_keypoints(
                )[self.current_frame][self.selected_keypoint]

                # Draw circle around selected keypoint
                processed_frame = cv2.circle(
                    copy.copy(frame), (int(keypoint_position[0]), int(keypoint_position[1])), 5, (255, 50, 50), 3)
            else:
                keypoint_position = None

                # If selected keypoint is not present on current frame it doesn't draw circle
                processed_frame = frame
        else:
            processed_frame = np.zeros((500, 500, 3)).astype(np.float32)
            keypoint_position = None

        # Additional plots
        if self.selected_plot == 0 and keypoint_position is not None:
            self._process_vector(
                frame, keypoint_position, fps)
            self.sc.axes.draw_artist(self.vector)  # Draw new vector
        elif self.selected_plot == 1:
            self.sc.axes.draw_artist(self.sc.axes.patch)

            # Gets all keypoints for current neural network
            keypoints = self._get_network_keypoints()

            # Gets only keypoints for current frame
            keypoints = keypoints[self.current_frame]

            # Process 2D frame
            self._process_2d(frame, keypoints)

            # Draw 2D skeleton
            self.sc.axes.draw_artist(self._2d_plot)
        # Update plot
        self.sc.fig.canvas.update()
        self.sc.fig.canvas.flush_events()

        # Color green available keypoints
        self._color_combobox_items()

        # Increment current frame value if possible
        if self.play_button_status and self.current_frame < len(self.networks[network]['Data'])-1:
            self.current_frame += 1

        if self.processing:
            return processed_frame, True

        return processed_frame, False

    # Function that receives processed frame from worker and draws it
    def _show_video_output(self, frame: np.array) -> None:
        # Sets max value for slider
        self.slider.setMaximum(self.number_of_frames-1)

        # Sets slider value to current frame value
        self.slider.setValue(self.current_frame)

        # Scale frame so it fits in window
        frame = utils.scale_frame(
            frame, self.settings.x_max_image_size, self.settings.y_max_image_size)

        # Draws processed frame
        self.image_window.set_image_data(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Changes vector values
    def set_vector(self, new_value: list) -> None:
        self.vector.set_UVC(new_value[0], new_value[1])

    # Changes play button status and text displayed on it
    def change_play_button(self) -> None:
        if self.play_button_status:
            self.play_button_status = False
            self.play_button.setText("PLAY")
        else:
            self.play_button_status = True
            self.play_button.setText("STOP")

    # Calculates the position derivative
    def calculate_new_vector(self, current_value: torch.Tensor, prev_keypoint_position: torch.Tensor, fps: float) -> list[float, float]:
        new_vector_x = (current_value[0] - prev_keypoint_position[0]) * fps
        new_vector_y = (current_value[1] - prev_keypoint_position[1]) * fps
        return [new_vector_x, new_vector_y]

    # Changes selected keypoint index
    def _selected_keypoint_change(self, i: str) -> None:
        self.selected_keypoint = int(i)

    # Changes selected plot index
    def _selected_plot_change(self, i: str) -> None:
        self.selected_plot = int(i)

    # Processes new vector and plot to display it
    def _process_vector(self, frame: np.array, keypoint_position: torch.tensor, fps: float) -> None:
        try:
            prev_keypoint_position = self._get_network_keypoints()[self.current_frame -
                                                                   1][self.selected_keypoint]
        except:
            prev_keypoint_position = torch.zeros((3, 1))

        # Reset plot
        self.sc.axes.draw_artist(self.sc.axes.patch)

        # Sets plot params for displaying vector
        self._set_vector_plot_params()

        # If keypoint position exists calculate new vector
        if keypoint_position is not None:
            # Calculate new vector
            new_vector_value = self.calculate_new_vector(
                keypoint_position, prev_keypoint_position, fps)

            # Set new vector
            self.set_vector(new_vector_value)

            # Save current position as previous
            self.pos_prev_value = keypoint_position

    # Function for configuring params of plt plot for vector display
    def _set_vector_plot_params(self) -> None:
        # Set axis limits
        self.sc.axes.set_xlim(-1000, 1000)
        self.sc.axes.set_ylim(-1000, 1000)

        # Set axis labels
        self.sc.axes.set_xlabel("px/s")
        self.sc.axes.set_ylabel("px/s")

        # Show visual components of x and y axis
        self.sc.axes.set_axis_on()

    # Process 2D skeleton display
    def _process_2d(self, frame: np.array, keypoints: torch.tensor) -> None:
        self._set_2d_plot_params(frame.shape)
        frame_y_shape = frame.shape[0]

        if len(keypoints) != 0:
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            print(f'Y: {y}, FRAME SHAPE: {frame.shape}')
            y = torch.tensor([frame_y_shape]*len(y)) - y
            print(f'NEW Y: {y}')
            self._2d_plot.set_offsets(np.c_[x, y])
        else:
            self._2d_plot.set_offsets(np.c_[0, 0])

    # Set params for displaying 2D plot
    def _set_2d_plot_params(self, frame_shape: tuple) -> None:
        # Set X and Y axis limits
        self.sc.axes.set_xlim(0, frame_shape[1])
        self.sc.axes.set_ylim(0, frame_shape[0])

        # Hide visual components of x and y axis
        self.sc.axes.set_axis_off()

    # Sets current frame from slider value
    def _set_current_frame(self, i: int) -> None:
        self.current_frame = i

    # Selects previous frame
    def _prev_frame(self) -> None:
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.setValue(self.current_frame)

    # Selects next frame
    def _next_frame(self) -> None:
        data = self._get_network_data()

        # Check if next frame is available
        if self.current_frame < len(data)-1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)

    # Returns currently selected network name
    def _get_network_name(self) -> str:
        network = list(self.networks.keys())[self.network_index]
        return network

    # Returns data for currently selected network
    def _get_network_data(self) -> list:
        network = self._get_network_name()
        data = self.networks[network]['Data']
        return data

    def _get_network_keypoints(self) -> list[torch.Tensor]:
        network = self._get_network_name()
        keypoints = self.networks[network]['Keypoints']
        return keypoints

    def _update_progress_bar(self, args: tuple) -> None:
        nn_name = args[0]  # Get name of the network
        value = args[1]  # Get value for the progress bar
        # Set progress bar value for one network
        self.networks[nn_name]['Progress_bar_value'] = value

        # Initialize variables for calculating final progress bar value
        number_of_processed_nns = 0
        total_progress_bar_value = 0

        # Get progress bar values for all processed networks to calculate the mean
        for network_name, values in self.networks.items():
            if values['Progress_bar_value'] > 0:
                number_of_processed_nns += 1
                total_progress_bar_value += values['Progress_bar_value']

        # Avoid division by 0
        if number_of_processed_nns > 0:
            # Set mean progress bar value
            self.progress_bar.setValue(
                int(total_progress_bar_value/number_of_processed_nns))

    # Function for coloring active keypoints
    def _color_combobox_items(self) -> None:
        # Get all keypoints for current NN
        keypoints = self._get_network_keypoints()

        # Avoid index out of range
        if len(keypoints) == 0:
            return

        # Get keypoints for current frame
        keypoints = keypoints[self.current_frame]

        # Get combobox model
        model = self.selected_keypoint_combobox.model()

        # Tensor for checking if keypoints are zeros
        zeros_tensor = torch.zeros((1, 3))

        # For each element in combobox set its color
        for i in range(TOTAL_NUMBER_OF_KEYPOINTS):
            # Check if keypoints exist and are non zero
            if keypoints.shape[0] == 0 or len(keypoints) <= i or torch.eq(keypoints[i], zeros_tensor).all():
                model.setData(model.index(i, 0), QtGui.QColor(
                    'gray'), QtCore.Qt.ItemDataRole.BackgroundRole)
            else:
                model.setData(model.index(i, 0), QtGui.QColor(
                    'green'), QtCore.Qt.ItemDataRole.BackgroundRole)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    app.exec()


if __name__ == '__main__':
    main()
