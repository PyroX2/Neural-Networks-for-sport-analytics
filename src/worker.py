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
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress_bar = pyqtSignal(tuple)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn  # Function to be called
        self.args = args  # Positional agruments
        self.kwargs = kwargs  # Key word arguments
        self.signals = WorkerSignals()  # Signals for sending results

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self.fn(
                *self.args, **self.kwargs, progress_bar_function=self.progress_bar_update
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

    def progress_bar_update(self, nn_name: str, value: int) -> None:
        self.signals.progress_bar.emit((nn_name, value))
