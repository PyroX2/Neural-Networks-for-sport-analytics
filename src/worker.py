'''
Implementation of Worker for MultiThreading
Multithreading is needed because of the time that
each neural network needs for processing the data
'''


class Worker(QRunnable):
    @pyqtSlot()
    def run(self):
        pass
