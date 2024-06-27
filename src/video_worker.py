'''
Implementation of Worker for MultiThreading
Multithreading is needed because of the time that
each neural network needs for processing the data
'''
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, pyqtSignal, QObject
import traceback
import sys


class WorkerSignals(QObject):
    result = pyqtSignal(object)


class VideoWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(VideoWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn  # Function to be called
        self.args = args  # Positional agruments
        self.kwargs = kwargs  # Key word arguments
        self.signals = WorkerSignals()  # Signals for sending results

    @pyqtSlot()
    def run(self):
        while True:
            frame, finished = self.fn(
                *self.args, **self.kwargs
            )
            self.signals.result.emit(frame)

            print(f'FINISHED: {finished}')
            if finished or finished is None:
                break
