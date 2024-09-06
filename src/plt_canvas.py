import sys
import time

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import random
import matplotlib.pyplot as plt
import src.utils as utils


class PltCanvas:
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()

        self.dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))

        self._dynamic_ax = self.dynamic_canvas.figure.subplots()

        self._change_plot(0)

        self.plot_index = 0

    def update_velocity_plot(self, vector: list) -> None:
        if hasattr(self, 'vector'):
            self._dynamic_ax.set_xlabel("px/s")
            self._dynamic_ax.set_ylabel("px/s")

            self.vector.set_UVC(vector[0], vector[1])
            self._dynamic_ax.set_ylim(-1, 1)
            self.vector.figure.canvas.draw()

    def update_2d_plot(self, keypoints: list, skeleton: list, frame_shape: tuple) -> None:
        self._dynamic_ax.set_xlim(0, frame_shape[1])
        self._dynamic_ax.set_ylim(0, frame_shape[0])

        lines = []

        print("KEYPOINTS", keypoints)
        if hasattr(self, '_line'):
            # Extract every point
            for i, (first_keypoint_index, second_keypoint_index) in enumerate(skeleton):
                print("first_keypoint_index", first_keypoint_index)
                start_point = keypoints[first_keypoint_index]
                end_point = keypoints[second_keypoint_index]
                if utils.keypoints_eq_to_zero(start_point, end_point):
                    continue

                line = self._dynamic_ax.plot([start_point[0], end_point[0]], [
                    frame_shape[0] - start_point[1], frame_shape[0] - end_point[1]])
                lines.append(line)

            # t = np.linspace(0, 10, 101)
            # self._line.set_data(t, np.sin(t + time.time()))
            for line in lines:
                line.figure.canvas.draw()

    def update_3d_plot(self):
        if hasattr(self, '_3d_line'):
            t = np.linspace(0, 10, 101)
            self._dynamic_ax.set_ylim(0, 10)
            self._dynamic_ax.set_zlim(-1, 1)
            self._dynamic_ax.set_xlim(0, 10)

            self._3d_line.set_data_3d(t, t, np.sin(t + time.time()))
            self._3d_line.figure.canvas.draw()

    def _update_canvas(self):
        if self.plot_index == 0:
            self.update_velocity_plot()
        elif self.plot_index == 1:
            self.update_2d_plot()
        else:
            self.update_3d_plot()

    def _change_plot(self, i: str) -> None:
        if i == 0:
            self._clear_plot()
            self._dynamic_ax = self.dynamic_canvas.figure.add_subplot(111)
            self._dynamic_ax.set_ylim(-1, 1)
            self.vector = self._dynamic_ax.quiver(
                0, 0, 10, 0)
        elif i == 1:
            self._clear_plot()
            self._dynamic_ax = self.dynamic_canvas.figure.add_subplot(111)
            self._dynamic_ax.set_ylim(-1, 1)
            self._line, = self._dynamic_ax.plot([0], [0])
        elif i == 2:
            self._clear_plot()
            self._dynamic_ax = self.dynamic_canvas.figure.add_subplot(111,
                                                                      projection='3d')
            self._3d_line, = self._dynamic_ax.plot(
                [0], [0], [0])
        self.plot_index = int(i)

    def _clear_plot(self):
        self._dynamic_ax.clear()
        self.dynamic_canvas.figure.delaxes(self._dynamic_ax)

    def _add_to_layout(self, layout, main_window):
        layout.addWidget(self.dynamic_canvas)
        layout.addWidget(NavigationToolbar(self.dynamic_canvas, main_window))


if __name__ == "__main__":
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = PltCanvas()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
