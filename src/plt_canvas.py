from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PltCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height),
                          dpi=dpi, facecolor='#2A2A2A')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#FFFFFF")
        self.axes.tick_params(axis='x', colors='#DDDDDD')
        self.axes.tick_params(axis='y', colors='#DDDDDD')
        super(PltCanvas, self).__init__(self.fig)
