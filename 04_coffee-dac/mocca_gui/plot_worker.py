# mocca_gui/plot_worker.py
#
# PlotWorker drives plotting entirely on the Qt main thread using a QTimer so
# that the event loop stays alive between bundles.  PyVista/VTK is not
# thread-safe; all add_mesh calls must happen on the main thread.

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import numpy as np
from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL


class PlotWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, plotter, edges_net, selection, show_endpoints):
        super().__init__()
        self.plotter = plotter
        self.edges_net = edges_net
        self.show_endpoints = show_endpoints
        self._cancelled = False

        # Expand "All" entries into individual (fcn, bundle) pairs up front
        self._bundle_queue = []
        for item in selection:
            fcn = item['fcn']
            if item['bundle'] == 'All':
                bundles = np.unique(
                    edges_net[edges_net[:, NETWORK_COL] == fcn][:, BUNDLE_COL]
                )
            else:
                bundles = [item['bundle']]
            for b in bundles:
                self._bundle_queue.append((fcn, b))

        self._total_bundles = max(len(self._bundle_queue), 1)
        self._done = 0

        self._timer = QTimer()
        self._timer.setInterval(0)   # fire as soon as the event loop is idle
        self._timer.timeout.connect(self._step)

    def start(self):
        self.plotter.clear()
        self._timer.start()

    def cancel(self):
        self._cancelled = True

    def _step(self):
        """Process one bundle per timer tick so the event loop stays responsive."""
        if self._cancelled or not self._bundle_queue:
            self._timer.stop()
            self.plotter.plotter.reset_camera()
            self.plotter.plotter.render()
            self.finished.emit()
            return

        fcn, b = self._bundle_queue.pop(0)

        edges = self.edges_net[
            (self.edges_net[:, NETWORK_COL] == fcn) &
            (self.edges_net[:, BUNDLE_COL] == b)
        ]

        if len(edges) == 0:
            self._done += 1
            self.progress.emit(int(self._done / self._total_bundles * 100))
            return

        # Resolve color
        from mocca_gui.colormap import my_colormap
        idx = self.plotter.bundle_colors.get((fcn, int(b)))
        if idx is None:
            idx = self.plotter.bundle_colors.get((fcn, 'All'))
        color = (my_colormap.colors[idx] if idx is not None
                 else my_colormap.colors[fcn % len(my_colormap.colors)])

        use_centroid = self.plotter.centroid_flags.get((fcn, int(b)), False)

        if use_centroid:
            from mocca_gui.plotter import generate_centroid_edge, plotline_ijk, add_endpoints_batch
            centroid_edge = generate_centroid_edge(edges)
            plotline_ijk(
                self.plotter.plotter,
                centroid_edge,
                color=color,
                offset_multiplier=self.plotter.curvatures.get((fcn, int(b)), 1.0),
                line_width=self.plotter.thicknesses.get((fcn, int(b)), 3),
                opacity=self.plotter.opacities.get((fcn, int(b)), 0.8),
            )
            if self.show_endpoints:
                # Single batched call: deduplicated unique endpoints across the
                # whole bundle, rendered as one glyph mesh
                add_endpoints_batch(
                    self.plotter.plotter,
                    edges,
                    color=color,
                    size_scale=self.plotter.endpoint_sizes.get((fcn, int(b)), 1.5),
                    opacity=self.plotter.opacities.get((fcn, int(b)), 0.8),
                )
        else:
            from mocca_gui.plotter import plotline_ijk, add_endpoints_batch
            for edge in edges:
                if self._cancelled:
                    break
                plotline_ijk(
                    self.plotter.plotter,
                    edge,
                    color=color,
                    offset_multiplier=self.plotter.curvatures.get((fcn, int(b)), 1.0),
                    line_width=self.plotter.thicknesses.get((fcn, int(b)), 3),
                    opacity=self.plotter.opacities.get((fcn, int(b)), 0.8),
                )
            # Draw all endpoints in one batched deduplicated call after the loop
            if self.show_endpoints and not self._cancelled:
                add_endpoints_batch(
                    self.plotter.plotter,
                    edges,
                    color=color,
                    size_scale=self.plotter.endpoint_sizes.get((fcn, int(b)), 1.5),
                    opacity=self.plotter.opacities.get((fcn, int(b)), 0.8),
                )

        self._done += 1
        self.progress.emit(int(self._done / self._total_bundles * 100))
