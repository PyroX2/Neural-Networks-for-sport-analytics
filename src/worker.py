'''
Implementation of Worker for MultiThreading
Multithreading is needed because of the time that
each neural network needs for processing the data
'''
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSlot, pyqtSignal, QObject
import traceback
import sys


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn  # Function to be called
        self.args = args  # Positional agruments
        self.kwargs = kwargs  # Key word arguments
        self.signals = WorkerSignals()  # Signals for sending results

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # Done
