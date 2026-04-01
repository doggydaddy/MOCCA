# mocca_gui/plot_worker.py

from PyQt5.QtCore import QThread, pyqtSignal

class PlotWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, plotter, edges_net, selection, show_endpoints):
        super().__init__()
        self.plotter = plotter
        self.edges_net = edges_net
        self.selection = selection
        self.show_endpoints = show_endpoints
        self._is_cancelled = False

    def run(self):
        # Run plotting and pass a progress callback
        self.plotter.draw_selection(
            self.edges_net,
            self.selection,
            endpoint_visible=self.show_endpoints,
            stop_flag=lambda: self._is_cancelled,
            progress_callback=self.progress.emit
        )
        self.plotter.draw_selection(..., progress_callback=self.progress.emit)
        self.finished.emit()

    def cancel(self):
        self._is_cancelled = True
