# mocca_gui/data_loader.py

import pandas as pd
from coffee_dac_pipeline import process_edge_data

from PyQt5.QtCore import QThread, pyqtSignal

class EdgeDataLoader:
    def __init__(self, main_window):
        self.main_window = main_window
        

    def load_edges(self, file_path):
        edges_net = process_edge_data(file_path)
        return edges_net

class EdgeDataLoaderWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        result = process_edge_data(
            self.file_path,
            progress_callback=self.progress.emit
        )
        self.finished.emit(result)