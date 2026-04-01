# mocca_gui/data_loader.py

import pandas as pd
from coffee_dac_pipeline import (
    process_edge_data,
    cache_exists,
    load_cached_result,
)
from coffee_dac_pipeline_v2 import (
    process_edge_data_v2,
    cache_exists_v2,
    load_cached_result_v2,
    recut_networks,
)

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

    def __init__(self, file_path, use_cache=False, pipeline='v1',
                 v2_kwargs=None, recut=None):
        '''
        Parameters
        ----------
        file_path  : str
        use_cache  : bool  – load from the appropriate cache rather than
                             running the full pipeline
        pipeline   : 'v1' | 'v2'
        v2_kwargs  : dict or None  – extra keyword arguments forwarded to
                     process_edge_data_v2 (e.g. top_n, min_network_size …)
                     Ignored when pipeline='v1' or use_cache=True.
        recut      : int or None  – if set and pipeline='v2', re-cut the
                     loaded result into this many networks using the cached
                     linkage matrix (instant, no reprocessing needed).
        '''
        super().__init__()
        self.file_path   = file_path
        self.use_cache   = use_cache
        self.pipeline    = pipeline
        self.v2_kwargs   = v2_kwargs or {}
        self.recut       = recut

    def run(self):
        if self.pipeline == 'v2':
            if self.use_cache:
                self.progress.emit(10)
                result = load_cached_result_v2(self.file_path)
                self.progress.emit(80)
                if self.recut is not None:
                    edges_out, nr_net = recut_networks(
                        result['edges_net'], result['linkage_matrix'], self.recut
                    )
                    result['edges_net'] = edges_out
                    result['nr_networks_out'] = nr_net
                self.progress.emit(100)
            else:
                result = process_edge_data_v2(
                    self.file_path,
                    progress_callback=self.progress.emit,
                    **self.v2_kwargs,
                )
        else:
            # v1 (original pipeline)
            if self.use_cache:
                self.progress.emit(10)
                result = load_cached_result(self.file_path)
                self.progress.emit(100)
            else:
                result = process_edge_data(
                    self.file_path,
                    progress_callback=self.progress.emit,
                )
        self.finished.emit(result)
