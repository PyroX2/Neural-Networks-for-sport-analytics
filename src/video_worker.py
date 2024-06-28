'''
Implementation of Worker for MultiThreading
Multithreading is needed because of the time that
each neural network needs for processing the data
'''
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, pyqtSignal, QObject
import traceback
import sys


# Worker Signals class for emiting signals outside of thread
class WorkerSignals(QObject):
    # Result for storing processed frame
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
            # Get processed frame
            frame, finished = self.fn(
                *self.args, **self.kwargs
            )

            # Send frame for displaying
            self.signals.result.emit(frame)

            # If processing returns finished break loop
            if finished or finished is None:
                break
