# mocca_gui/main_window.py
# mocca_gui/main_window.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QPushButton,
    QHBoxLayout, QLabel, QCheckBox, QSlider, QFileDialog, QMessageBox,
    QProgressDialog, QDialog, QVBoxLayout, QComboBox, QDialogButtonBox,
    QColorDialog, QSpinBox, QApplication, QFormLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor

from mocca_gui.plotter import NetworkPlotter
from mocca_gui.tree_manager import TreeManager
from mocca_gui.fine_tuner import FineTuner
from mocca_gui.data_loader import EdgeDataLoader
from mocca_gui.gif_exporter import GifExporter
from mocca_gui.figure_exporter import FigureExporter
from mocca_gui.publication_exporter import (
    COLOR_MODE_GUI,
    COLOR_MODE_PUBLICATION,
    PublicationExporter,
    ExportCancelled,
)
from mocca_gui.dendrogram_plotter import show_dendrogram
from mocca_gui.plot_worker import PlotWorker
from mocca_gui.data_loader import EdgeDataLoaderWorker

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL

import pyvista as pv
import math
import os
import numpy as np

from mocca_gui import colormap

# Ensure colors are tuples
colormap.my_colormap.colors = [
    tuple(c.tolist()) if hasattr(c, "tolist") else tuple(c)
    for c in colormap.my_colormap.colors
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COFFEE-DAC FCN & Bundle Viewer")
        self.resize(1200, 900)

        self.edges_net = None

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Splitter
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # PyVista plotter
        self.plotter_widget = QtInteractor()
        self.plotter = NetworkPlotter(self.plotter_widget.interactor)
        splitter.addWidget(self.plotter_widget)

        # Tree manager
        self.tree_manager = TreeManager(self)
        splitter.addWidget(self.tree_manager.widget)

        # Bottom control panel
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)

        # Top row of buttons
        controls_layout = QHBoxLayout()

        # Load Data button
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_data_dialog)
        controls_layout.addWidget(load_btn)

        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self.show_all)
        controls_layout.addWidget(show_all_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_plot)
        controls_layout.addWidget(clear_btn)

        toggle_centroids_btn = QPushButton("Toggle All Centroids")
        toggle_centroids_btn.clicked.connect(self.toggle_all_centroids_global)
        controls_layout.addWidget(toggle_centroids_btn)

        dendro_btn = QPushButton("Show Dendrogram")
        dendro_btn.clicked.connect(self.show_dendrogram)
        controls_layout.addWidget(dendro_btn)


        self.endpoint_checkbox = QCheckBox("Show Endpoints")
        self.endpoint_checkbox.setChecked(True)
        controls_layout.addWidget(self.endpoint_checkbox)

        plot_btn = QPushButton("Plot Selection")
        plot_btn.clicked.connect(self.plot_selected)
        controls_layout.addWidget(plot_btn)

        export_btn = QPushButton("Export GIF")
        export_btn.clicked.connect(self.export_gif_dialog)
        controls_layout.addWidget(export_btn)

        export_figure_btn = QPushButton("Export Figure")
        export_figure_btn.setToolTip(
            "Export the current camera view at publication resolution with "
            "caption and provenance metadata"
        )
        export_figure_btn.clicked.connect(self.export_figure_dialog)
        controls_layout.addWidget(export_figure_btn)

        export_all_btn = QPushButton("Export All GIFs")
        export_all_btn.clicked.connect(self.export_all_gifs)
        controls_layout.addWidget(export_all_btn)

        fine_all_btn = QPushButton("Fine Tune All FCNs")
        fine_all_btn.clicked.connect(self.fine_tune_all_fcns)
        controls_layout.addWidget(fine_all_btn)

        bottom_layout.addLayout(controls_layout)

        # Fine tuning panel
        self.fine_tuner = FineTuner(self)
        bottom_layout.addWidget(self.fine_tuner.widget)

        # GIF settings panel
        gif_panel = self.build_gif_settings_panel()
        bottom_layout.addWidget(gif_panel)

        # Brain mesh opacity slider
        brain_opacity_panel = self.build_brain_opacity_panel()
        bottom_layout.addWidget(brain_opacity_panel)

        layout.addWidget(bottom_panel)

        self.data_loader = EdgeDataLoader(self)
        self.gif_exporter = GifExporter()
        self.figure_exporter = FigureExporter()
        self.publication_exporter = PublicationExporter()
        self.loaded_data_path = None
        self.loaded_pipeline = None
        self.loaded_provenance = None

        splitter.setSizes([700, 300])

        self.preview_timer = None
        self.preview_frame = 0
    
    def load_data_dialog(self):
        from mocca_gui.data_loader import EdgeDataLoaderWorker
        from coffee_dac_pipeline import cache_exists
        from coffee_dac_pipeline_v2 import (
            cache_exists_v2,
            cache_validation_v2,
            is_processed_input_v2,
            load_params_v2,
        )
        from coffee_dac_pipeline_v3 import (
            cache_exists_v3,
            cache_validation_v3,
            is_processed_input_v3,
            load_params_v3,
        )

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Edge CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        if is_processed_input_v2(path):
            QMessageBox.warning(
                self,
                "Processed cache selected",
                "This is a v2 processed cache, not a raw input CSV.\n\n"
                "Select the corresponding original CSV instead. Loading a "
                "processed cache as input would create nested names such as "
                "'_v2_processed_v2_processed.csv'.",
            )
            return

        if is_processed_input_v3(path):
            QMessageBox.warning(
                self,
                "Processed cache selected",
                "This is a v3 (divisive) processed cache, not a raw input "
                "CSV.\n\nSelect the corresponding original CSV instead.",
            )
            return

        has_v1 = cache_exists(path)
        has_v2 = cache_exists_v2(path)
        has_v3 = cache_exists_v3(path)
        v2_manifest = load_params_v2(path) if has_v2 else None
        v2_cache_valid, v2_cache_reason = (
            cache_validation_v2(path) if has_v2 else (False, None)
        )
        v3_manifest = load_params_v3(path) if has_v3 else None
        v3_cache_valid, v3_cache_reason = (
            cache_validation_v3(path) if has_v3 else (False, None)
        )

        # --- Build the prompt dialog when any cache exists ---
        pipeline = 'v1'
        use_cache = False
        recut = None

        if has_v1 or has_v2 or has_v3:
            dialog = QDialog(self)
            dialog.setWindowTitle("Cached results found")
            layout = QVBoxLayout(dialog)

            cache_info = []
            if has_v3:
                cache_info.append("  • v3 cache (divisive: sub-bundles + edge-level linkage)")
                if v3_manifest is not None:
                    params = v3_manifest.get('parameters') or {}
                    results = v3_manifest.get('results') or {}
                    parent_bundles = results.get('parent_bundles') or []
                    parent_lines = "\n".join(
                        f"      parent {entry.get('parent_bundle_id', '?')}: "
                        f"{entry.get('edge_count', '?')} edges -> "
                        f"{entry.get('subdivisions', '?')} sub-bundle(s)"
                        for entry in parent_bundles
                    )
                    cache_info.append(
                        "    Recorded parameters: "
                        f"nr-bundles={params.get('nr_bundles', '?')}, "
                        f"h1-flag={params.get('h1_flag', '?')}, "
                        f"method={params.get('method', '?')}\n"
                        "    Result: "
                        f"{results.get('retained_edges', '?')} edges, "
                        f"{results.get('parent_bundle_count', '?')} parent bundle(s)\n"
                        + (parent_lines + "\n" if parent_lines else "")
                        + "    Completed: "
                        f"{v3_manifest.get('completed_at', 'unknown')}"
                    )
                else:
                    cache_info.append(
                        "    Parameter metadata unavailable (legacy cache)"
                    )
                if not v3_cache_valid and v3_manifest is not None:
                    cache_info.append(
                        f"    Warning: cache validation failed: {v3_cache_reason}"
                    )
            if has_v2:
                cache_info.append("  • v2 cache (processed CSV + linkage matrix)")
                if v2_manifest is not None:
                    params = v2_manifest.get('parameters') or {}
                    results = v2_manifest.get('results') or {}
                    cache_info.append(
                        "    Recorded parameters: "
                        f"networks={params.get('nr_networks', '?')}, "
                        f"min-size={params.get('min_network_size', '?')}, "
                        "min-cluster-voxels="
                        f"{params.get('min_cluster_voxels', '?')}, "
                        f"neighbor-dist={params.get('neighbor_dist', '?')}, "
                        f"strict-bundles={params.get('strict_bundles', '?')}, "
                        f"top-n={params.get('top_n')}, "
                        f"tstat-threshold={params.get('tstat_threshold')}\n"
                        "    Result: "
                        f"{results.get('retained_edges', '?')} edges, "
                        f"{results.get('bundles', '?')} bundles, "
                        f"{results.get('networks', '?')} networks\n"
                        "    Completed: "
                        f"{v2_manifest.get('completed_at', 'unknown')}"
                    )
                else:
                    cache_info.append(
                        "    Parameter metadata unavailable (legacy cache)"
                    )
                if not v2_cache_valid and v2_manifest is not None:
                    cache_info.append(
                        f"    Warning: cache validation failed: {v2_cache_reason}"
                    )
            if has_v1:
                cache_info.append("  • v1 cache (processed CSV + linkage matrix)")
            layout.addWidget(QLabel(
                "Previously processed results were found for this dataset:\n" +
                "\n".join(cache_info) +
                "\n\nHow would you like to proceed?"
            ))

            combo = QComboBox(dialog)
            if has_v3:
                combo.addItem("Load existing v3 results — divisive (fast)", ("v3", True))
            if has_v2:
                combo.addItem("Load existing v2 results (fast)", ("v2", True))
            if has_v1:
                combo.addItem("Load existing v1 results (fast)", ("v1", True))
            combo.addItem("Re-process with pipeline v3 — divisive (slow)", ("v3", False))
            combo.addItem("Re-process with pipeline v2 (slow)", ("v2", False))
            combo.addItem("Re-process with pipeline v1 (slow)", ("v1", False))
            layout.addWidget(combo)

            # Recut controls — relevant when loading a v2 (networks) or v3
            # (sub-bundles) cache. v2 cuts the whole cached tree at one size;
            # v3 cuts one INDEPENDENT tree per parent (inferential) bundle,
            # so each parent bundle gets its own spinbox -- a small parent
            # bundle may only need 2 sub-bundles to read clearly while a
            # much larger one needs 6 (see coffee_dac_pipeline_v3.py).
            recut_widget = QWidget(dialog)
            recut_layout = QHBoxLayout(recut_widget)
            recut_layout.setContentsMargins(0, 0, 0, 0)
            recut_label = QLabel("Cut into N networks (v2 cache only):")
            recut_spin = QSpinBox(dialog)
            recut_spin.setRange(2, 50)
            recorded_networks = (
                (v2_manifest or {}).get('results', {}).get('networks', 5)
            )
            try:
                recorded_networks = int(recorded_networks)
            except (TypeError, ValueError):
                recorded_networks = 5
            recut_spin.setValue(max(2, min(50, recorded_networks)))
            recut_layout.addWidget(recut_label)
            recut_layout.addWidget(recut_spin)
            layout.addWidget(recut_widget)

            recorded_parent_bundles = (
                (v3_manifest or {}).get('results', {}).get('parent_bundles') or []
            )
            v3_recut_widget = QWidget(dialog)
            v3_recut_form = QFormLayout(v3_recut_widget)
            v3_recut_spins = {}
            if recorded_parent_bundles:
                for entry in recorded_parent_bundles:
                    parent_id = int(entry.get('parent_bundle_id', 0))
                    try:
                        recorded_n = max(2, min(50, int(entry.get('subdivisions', 2))))
                    except (TypeError, ValueError):
                        recorded_n = 2
                    spin = QSpinBox(dialog)
                    spin.setRange(1, 50)
                    spin.setValue(recorded_n)
                    v3_recut_spins[parent_id] = spin
                    v3_recut_form.addRow(
                        f"Parent bundle {parent_id} "
                        f"({entry.get('edge_count', '?')} edges) -> N sub-bundles:",
                        spin,
                    )
            else:
                # Legacy cache with no per-parent breakdown recorded: fall
                # back to one spinbox applied uniformly to every parent
                # bundle present (recut_subbundles accepts a plain int).
                spin = QSpinBox(dialog)
                spin.setRange(2, 50)
                spin.setValue(2)
                v3_recut_spins[None] = spin
                v3_recut_form.addRow("Cut every bundle into N sub-bundles:", spin)
            v3_recut_scroll = QScrollArea(dialog)
            v3_recut_scroll.setWidget(v3_recut_widget)
            v3_recut_scroll.setWidgetResizable(True)
            v3_recut_scroll.setMaximumHeight(200)
            layout.addWidget(v3_recut_scroll)

            def update_recut_visibility():
                choice_pipeline, choice_cache = combo.currentData()
                if choice_pipeline == 'v3' and choice_cache:
                    recut_widget.setVisible(False)
                    v3_recut_scroll.setVisible(True)
                elif choice_pipeline == 'v2' and choice_cache:
                    recut_label.setText("Cut into N networks (v2 cache only):")
                    recut_spin.setValue(max(2, min(50, recorded_networks)))
                    recut_widget.setVisible(True)
                    v3_recut_scroll.setVisible(False)
                else:
                    recut_widget.setVisible(False)
                    v3_recut_scroll.setVisible(False)
            combo.currentIndexChanged.connect(update_recut_visibility)
            update_recut_visibility()

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(buttons)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            if dialog.exec_() != QDialog.Accepted:
                return

            pipeline, use_cache = combo.currentData()
            if pipeline == 'v2' and use_cache:
                recut = recut_spin.value()
            elif pipeline == 'v3' and use_cache:
                if None in v3_recut_spins:
                    recut = v3_recut_spins[None].value()
                else:
                    recut = {
                        parent_id: spin.value()
                        for parent_id, spin in v3_recut_spins.items()
                    }
        else:
            # No cache at all — default to v2 processing
            pipeline = 'v2'
            use_cache = False

        self.progress_dialog = QProgressDialog(
            "Loading data...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)

        self.loader_worker = EdgeDataLoaderWorker(
            path,
            use_cache=use_cache,
            pipeline=pipeline,
            recut=recut,
        )

        self.loader_worker.progress.connect(self.progress_dialog.setValue)
        self.loader_worker.finished.connect(self.on_data_loaded)
        self.progress_dialog.canceled.connect(self.loader_worker.terminate)

        self.loader_worker.start()
        self.progress_dialog.show()

    def on_data_loaded(self, result):
        self.edges_net = result['edges_net']
        self.loaded_data_path = result.get('input_path')
        self.loaded_pipeline = result.get('pipeline')
        self.loaded_provenance = result.get('provenance')
        self.linkage_matrix = result.get('linkage_matrix')
        # v3: one independent linkage tree PER PARENT bundle (see
        # coffee_dac_pipeline_v3.py), keyed by parent (NETWORK_COL) id --
        # never a single tree spanning every parent bundle.
        self.linkage_matrices = result.get('linkage_matrices')
        # v3 (divisive) builds one linkage leaf per EDGE, since it clusters
        # individual edges directly rather than bundles; v1/v2 build one
        # leaf per BUNDLE. The dendrogram view needs to know which, since it
        # can't tell reliably just from array shapes.
        self.dendrogram_leaves = 'edges' if result.get('pipeline') == 'v3' else 'bundles'
        self.tree_manager.populate(self.edges_net)
        self.plotter.clear()
        self.progress_dialog.close()

    def show_all(self):
        if self.edges_net is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        selection = []
        fcn_ids = sorted(set(self.edges_net[:,NETWORK_COL].astype(int)))
        for f in fcn_ids:
            selection.append({"fcn": f, "bundle": "All"})
        self.plotter.draw_selection(
            self.edges_net, selection, self.endpoint_checkbox.isChecked()
        )

    def clear_plot(self):
        self.plotter.clear()
        self.tree_manager.clear_selection()

    def plot_selected(self):
        selection = self.tree_manager.get_selection()
        if not selection:
            QMessageBox.information(self, "Nothing selected", "Select bundles or FCNs first.")
            return

        show_eps = self.endpoint_checkbox.isChecked()

        # set up progress dialog
        self.progress_dialog = QProgressDialog(
            "Plotting...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        cancelled_flag = [False]

        def cancel_plot():
            cancelled_flag[0] = True

        # QProgressDialog does not stop synchronous work by itself.  Record the
        # cancellation so draw_selection can leave its rendering loops.
        self.progress_dialog.canceled.connect(cancel_plot)

        self.plotter.draw_selection(
            self.edges_net,
            selection,
            endpoint_visible=show_eps,
            stop_flag=lambda: cancelled_flag[0],
            progress_callback=self.progress_dialog.setValue
        )

        self.progress_dialog.close()

    def choose_color(self, fcn, bundle, button_widget):
        from mocca_gui.colormap import my_colormap

        # Get all colors from your custom colormap
        color_list = my_colormap.colors

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Choose Color for FCN {fcn}, Bundle {bundle}")

        layout = QVBoxLayout(dialog)

        combo = QComboBox(dialog)
        for i, rgba in enumerate(color_list):
            icon = self.create_color_icon(rgba)
            label = f"Color {i+1}"
            if i == fcn % len(color_list):  # Mark default FCN color
                label += " (default)"
            combo.addItem(icon, label, i)

        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            idx = combo.currentData()
            self.plotter.bundle_colors[(fcn, bundle)] = idx

            rgba = color_list[idx]
            button_widget.setStyleSheet(
                f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]});"
            )

            self.plot_selected()   

    def reset_bundle_color(self, fcn, bundle_id):
        """
        Resets the color of a bundle to inherit from the FCN color.
        """
        from mocca_gui.colormap import my_colormap

        # Remove custom bundle color
        self.plotter.bundle_colors[(fcn, bundle_id)] = None
        print(f"Bundle {bundle_id} in FCN {fcn} reset to FCN color.")

        # Check for custom FCN color stored under (fcn, "All")
        custom_fcn_color_idx = self.plotter.bundle_colors.get((fcn, "All"), None)

        if custom_fcn_color_idx is not None:
            rgba = my_colormap.colors[custom_fcn_color_idx]
        else:
            rgba = my_colormap.colors[fcn]

        rgba_tuple = (
            tuple(rgba.tolist())
            if hasattr(rgba, "tolist")
            else tuple(rgba)
        )
        r, g, b, a = rgba_tuple

        # Update the color button
        btn = self.tree_manager.bundle_color_buttons.get((fcn, bundle_id))
        if btn:
            btn.setStyleSheet(
                f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 255);"
            )

        self.plot_selected()

    def choose_fcn_color(self, fcn, button_widget):
        from mocca_gui.colormap import my_colormap

        color_list = my_colormap.colors

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Choose Color for FCN {fcn} (All Bundles)")

        layout = QVBoxLayout(dialog)

        combo = QComboBox(dialog)
        for i, rgba in enumerate(color_list):
            icon = self.create_color_icon(rgba)
            label = f"Color {i+1}"
            if i == fcn % len(color_list):  # Mark default FCN color
                label += " (default)"
            combo.addItem(icon, label, i)


        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            idx = combo.currentData()

            self.plotter.bundle_colors[(fcn, 'All')] = idx

            rgba = color_list[idx]
            button_widget.setStyleSheet(
                f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]});"
            )

            self.plot_selected()

    def reset_fcn_color(self, fcn):
        """
        Resets the FCN color override to the default color,
        and resets all bundles in that FCN to show the default color.
        """
        from mocca_gui.colormap import my_colormap
    
        # Remove custom FCN color override entirely
        key = (fcn, "All")
        if key in self.plotter.bundle_colors:
            del self.plotter.bundle_colors[key]
            print(f"Removed custom color override for FCN {fcn}.")
        else:
            print(f"No custom color override existed for FCN {fcn}.")
    
        # Get the default color for the FCN
        rgba = my_colormap.colors[fcn]
        rgba_tuple = (
            tuple(rgba.tolist())
            if hasattr(rgba, "tolist")
            else tuple(rgba)
        )
        r, g, b, a = rgba_tuple
    
        # Update the "All" button
        all_btn = self.tree_manager.bundle_color_buttons.get((fcn, "All"))
        if all_btn:
            all_btn.setStyleSheet(
                f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 255);"
            )
            all_btn.repaint()
    
        # Reset all bundles in this FCN
        for (fcn_candidate, bundle_candidate), _ in list(self.plotter.bundle_colors.items()):
            if fcn_candidate == fcn and bundle_candidate != "All":
                del self.plotter.bundle_colors[(fcn_candidate, bundle_candidate)]
                print(f"Reset bundle {bundle_candidate} in FCN {fcn} to default FCN color.")
    
                btn = self.tree_manager.bundle_color_buttons.get((fcn_candidate, bundle_candidate))
                if btn:
                    btn.setStyleSheet(
                        f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 255);"
                    )
                    btn.repaint()
    
        # Also update bundles that never had custom colors
        for (fcn_candidate, bundle_candidate), btn in self.tree_manager.bundle_color_buttons.items():
            if fcn_candidate == fcn and bundle_candidate != "All":
                btn.setStyleSheet(
                    f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 255);"
                )
                btn.repaint()
    
        self.plot_selected()

    def create_color_icon(self, rgba, size=16):
        pixmap = QPixmap(int(size), int(size))
        color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        pixmap.fill(color)
        return QIcon(pixmap)

    def toggle_centroid(self, fcn, bundle, button_widget):
        key = (fcn, bundle)
        current = self.plotter.centroid_flags.get(key, False)
        new_state = not current
        self.plotter.centroid_flags[key] = new_state

        if new_state:
            button_widget.setText("Centroid ✓")
        else:
            button_widget.setText("Centroid")

        self.plot_selected()

    def toggle_all_centroids(self, fcn):
        # Determine if any bundle currently has centroid enabled
        all_on = any(
            self.plotter.centroid_flags.get((fcn, b), False)
            for b in np.unique(self.edges_net[self.edges_net[:,NETWORK_COL] == fcn][:,BUNDLE_COL])
        )

        new_state = not all_on

        for b in np.unique(self.edges_net[self.edges_net[:,NETWORK_COL] == fcn][:,BUNDLE_COL]):
            self.plotter.centroid_flags[(fcn, int(b))] = new_state

        # Update centroid button labels via the stored button references
        for b in np.unique(self.edges_net[self.edges_net[:,NETWORK_COL] == fcn][:,BUNDLE_COL]):
            btn = self.tree_manager.bundle_centroid_buttons.get((fcn, int(b)))
            if btn:
                btn.setText("Centroid ✓" if new_state else "Centroid")

        self.plot_selected()

    def toggle_all_centroids_global(self):
        if self.edges_net is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return

        # Determine current global state (are any centroids on?)
        any_on = any(self.plotter.centroid_flags.get((int(f), int(b)), False)
                    for f in np.unique(self.edges_net[:,NETWORK_COL])
                    for b in np.unique(self.edges_net[self.edges_net[:,NETWORK_COL]==f][:,BUNDLE_COL]))

        new_state = not any_on  # Toggle

        for f in np.unique(self.edges_net[:,NETWORK_COL]):
            for b in np.unique(self.edges_net[self.edges_net[:,NETWORK_COL]==f][:,BUNDLE_COL]):
                self.plotter.centroid_flags[(int(f), int(b))] = new_state

        # Update centroid button labels via the stored button references
        for (f, b), btn in self.tree_manager.bundle_centroid_buttons.items():
            btn.setText("Centroid ✓" if new_state else "Centroid")

        self.plot_selected()
    
    def export_gif_dialog(self):
        selection = self.tree_manager.get_selection()
        if not selection:
            QMessageBox.warning(self, "Nothing selected", "Select something first!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF Files (*.gif)")
        if filename:
            self.gif_exporter.export(
                edges_net=self.edges_net,
                selection=selection,
                filename=filename,
                elevation=self.elevation_slider.value(),
                azimuth=self.azimuth_slider.value(),
                plotter=self.plotter,
                endpoint_visible=self.endpoint_checkbox.isChecked(),
            )
            QMessageBox.information(self, "Done", f"GIF saved:\n{filename}")

    def export_figure_dialog(self):
        if (
            self.edges_net is None
            or not self.plotter._edge_actors
            or not self.plotter.last_selection
        ):
            QMessageBox.warning(
                self, "Nothing plotted", "Plot a selection before exporting a figure."
            )
            return

        chooser = QMessageBox(self)
        chooser.setWindowTitle("Export Figure")
        chooser.setIcon(QMessageBox.Information)
        chooser.setText("Choose the figure export type.")
        chooser.setInformativeText(
            "Publication Set creates standardized lateral/dorsal views, an "
            "endpoint-density panel, full-edge supplements, summary CSV, "
            "captions, and a reproducibility manifest."
        )
        publication_button = chooser.addButton(
            "Publication Set", QMessageBox.AcceptRole
        )
        current_button = chooser.addButton(
            "Current View", QMessageBox.ActionRole
        )
        chooser.addButton(QMessageBox.Cancel)
        chooser.setDefaultButton(publication_button)
        chooser.exec_()

        if chooser.clickedButton() is publication_button:
            self.export_publication_set_dialog()
        elif chooser.clickedButton() is current_button:
            self.export_current_view_dialog()

    def export_current_view_dialog(self):

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Publication Figure",
            "publication_figure.png",
            (
                "PNG Figure (*.png);;TIFF Figure (*.tif *.tiff);;"
                "PDF Figure (*.pdf);;SVG Figure (*.svg)"
            ),
        )
        if not filename:
            return
        if not os.path.splitext(filename)[1]:
            extension_by_filter = {
                "PNG Figure (*.png)": ".png",
                "TIFF Figure (*.tif *.tiff)": ".tif",
                "PDF Figure (*.pdf)": ".pdf",
                "SVG Figure (*.svg)": ".svg",
            }
            filename += extension_by_filter.get(selected_filter, ".png")

        try:
            outputs = self.figure_exporter.export(
                filename=filename,
                network_plotter=self.plotter,
                edges_net=self.edges_net,
                pipeline=self.loaded_pipeline,
                input_path=self.loaded_data_path,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Figure export failed", f"Could not export the figure:\n{exc}"
            )
            return

        QMessageBox.information(
            self,
            "Figure Exported",
            "Publication figure and sidecars saved:\n\n"
            f"Figure: {outputs['figure']}\n"
            f"Caption: {outputs['caption']}\n"
            f"Metadata: {outputs['metadata']}",
        )

    def export_publication_set_dialog(self):
        palette_chooser = QMessageBox(self)
        palette_chooser.setWindowTitle("Publication Set Colors")
        palette_chooser.setIcon(QMessageBox.Question)
        palette_chooser.setText("Choose how bundle and network colors are exported.")
        palette_chooser.setInformativeText(
            "Current GUI colors preserves the colors shown in the viewer and "
            "GIF exports. Standardized palette uses the color-vision-deficiency-"
            "safe Okabe-Ito publication palette."
        )
        gui_colors_button = palette_chooser.addButton(
            "Current GUI Colors", QMessageBox.AcceptRole
        )
        publication_colors_button = palette_chooser.addButton(
            "Standardized Palette", QMessageBox.ActionRole
        )
        palette_chooser.addButton(QMessageBox.Cancel)
        palette_chooser.setDefaultButton(gui_colors_button)
        palette_chooser.exec_()

        if palette_chooser.clickedButton() is gui_colors_button:
            color_mode = COLOR_MODE_GUI
        elif palette_chooser.clickedButton() is publication_colors_button:
            color_mode = COLOR_MODE_PUBLICATION
        else:
            return

        parent_directory = QFileDialog.getExistingDirectory(
            self,
            "Select folder for the new timestamped publication export",
        )
        if not parent_directory:
            return

        progress = QProgressDialog(
            "Creating standardized publication figures...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Publication Export")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        cancelled = [False]
        progress.canceled.connect(lambda: cancelled.__setitem__(0, True))

        def update_progress(value):
            progress.setValue(value)
            QApplication.processEvents()

        try:
            outputs = self.publication_exporter.export(
                output_parent=parent_directory,
                network_plotter=self.plotter,
                edges_net=self.edges_net,
                pipeline=self.loaded_pipeline,
                input_path=self.loaded_data_path,
                provenance=self.loaded_provenance,
                progress_callback=update_progress,
                stop_flag=lambda: cancelled[0],
                color_mode=color_mode,
            )
        except ExportCancelled:
            QMessageBox.information(
                self, "Export Cancelled", "No publication export folder was created."
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Publication export failed",
                f"Could not create the publication export set:\n{exc}",
            )
            return
        finally:
            progress.close()

        QMessageBox.information(
            self,
            "Publication Set Exported",
            "Publication export created:\n\n"
            f"Folder: {outputs['directory']}\n"
            f"Main figure: {outputs['figure_png']}\n"
            f"PDF: {outputs['figure_pdf']}\n"
            f"Summary: {outputs['summary']}\n"
            f"Manifest: {outputs['manifest']}\n"
            f"Colors: {'current GUI colors' if color_mode == COLOR_MODE_GUI else 'standardized palette'}",
        )

    def export_all_gifs(self):
        # Placeholder for your export_all_networks code
        dir_path = QFileDialog.getExistingDirectory(self, "Select folder to save GIFs")
        if dir_path:
            QMessageBox.information(self, "Export All", "All GIFs exported. (Stub)")

    def fine_tune_all_fcns(self):
        self.fine_tuner.load_target("global")

    def fine_tune_bundle(self, fcn, bundle):
        self.fine_tuner.load_target("bundle", fcn, bundle)

    def fine_tune_fcn(self, fcn):
        self.fine_tuner.load_target("fcn", fcn)

    # ---------------- GIF panel ------------------------

    def build_gif_settings_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)

        label = QLabel("GIF Rotation:")
        layout.addWidget(label)

        self.elevation_label = QLabel("Elevation: 0°")
        layout.addWidget(self.elevation_label)

        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-60, 60)
        self.elevation_slider.setValue(0)
        self.elevation_slider.valueChanged.connect(
            lambda val: self.elevation_label.setText(f"Elevation: {val}°"))
        layout.addWidget(self.elevation_slider)

        layout.addSpacing(20)

        self.azimuth_label = QLabel("Azimuth: 0°")
        layout.addWidget(self.azimuth_label)

        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(0, 360)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.valueChanged.connect(
            lambda val: self.azimuth_label.setText(f"Azimuth: {val}°"))
        layout.addWidget(self.azimuth_slider)

        self.preview_checkbox = QCheckBox("Live Preview")
        self.preview_checkbox.stateChanged.connect(self.toggle_preview)
        layout.addWidget(self.preview_checkbox)

        return panel

    def update_preview(self):
        self.preview_frame += 1

        # Compute absolute angles to match GIF logic
        absolute_azimuth = self.azimuth_slider.value() + self.preview_frame * 2
        absolute_elevation = self.elevation_slider.value()

        # Restore initial camera position
        self.plotter_widget.interactor.camera_position = self.preview_initial_camera_position

        # Rotate the camera to desired angles
        camera = self.plotter_widget.interactor.camera

        # Apply absolute azimuth
        camera.Azimuth(absolute_azimuth)
        # Apply absolute elevation
        camera.Elevation(absolute_elevation)

        self.plotter_widget.interactor.render()

    def toggle_preview(self, checked):
        if checked:
            self.preview_frame = 0

            # Save initial camera position
            self.preview_initial_camera_position = self.plotter_widget.interactor.camera_position

            self.preview_timer = QTimer()
            self.preview_timer.timeout.connect(self.update_preview)
            self.preview_timer.start(50)
        else:
            if self.preview_timer:
                self.preview_timer.stop()
                self.preview_timer = None
            self.plotter_widget.interactor.reset_camera()
            self.plotter_widget.interactor.render()

    def get_camera_position(self, frame=0, total_frames=180):
        base_azimuth = self.azimuth_slider.value()
        base_elevation = self.elevation_slider.value()
        azimuth = base_azimuth + (frame * 360 / total_frames)
        oscillation = 5 * math.sin(frame * 2 * math.pi / 180)
        elevation = base_elevation + oscillation
        return azimuth, elevation

    # ---------------- Brain opacity sliders ------------------------

    def build_brain_opacity_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # GM slider
        layout.addWidget(QLabel("GM Opacity:"))
        self.gm_opacity_label = QLabel("35%")
        layout.addWidget(self.gm_opacity_label)
        self.gm_opacity_slider = QSlider(Qt.Horizontal)
        self.gm_opacity_slider.setRange(0, 100)
        self.gm_opacity_slider.setValue(35)
        self.gm_opacity_slider.setTickInterval(10)
        self.gm_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.gm_opacity_slider.valueChanged.connect(self.on_gm_opacity_changed)
        layout.addWidget(self.gm_opacity_slider)

        layout.addSpacing(20)

        # WM slider
        layout.addWidget(QLabel("WM Opacity:"))
        self.wm_opacity_label = QLabel("10%")
        layout.addWidget(self.wm_opacity_label)
        self.wm_opacity_slider = QSlider(Qt.Horizontal)
        self.wm_opacity_slider.setRange(0, 100)
        self.wm_opacity_slider.setValue(10)
        self.wm_opacity_slider.setTickInterval(10)
        self.wm_opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.wm_opacity_slider.valueChanged.connect(self.on_wm_opacity_changed)
        layout.addWidget(self.wm_opacity_slider)

        return panel

    def on_gm_opacity_changed(self, value):
        self.gm_opacity_label.setText(f"{value}%")
        self.plotter.set_layer_opacity('gm', value / 100.0)

    def on_wm_opacity_changed(self, value):
        self.wm_opacity_label.setText(f"{value}%")
        self.plotter.set_layer_opacity('wm', value / 100.0)

    # ---------------- Dendrogram Plotting ------------------------

    def _prepare_dendrogram_panel(self, edges_net, Z, per_edge_leaves, title):
        import numpy as np
        from mocca_gui.colormap import my_colormap

        # get number of FCNs present in this panel's edges
        num_fcns = len(np.unique(edges_net[:, NETWORK_COL]))

        def find_nth_largest_link(Z, n):
            # Z[:, 2] contains the distances of the merges
            distances = Z[:, 2]
            sorted_distances = np.sort(distances)[::-1]  # descending order
            if n > len(sorted_distances):
                return None
            nth_distance = sorted_distances[n - 1]
            return nth_distance

        # Get unique bundles (for v3 these are the sub-bundles of ONE parent
        # bundle -- this panel's edges already belong to a single parent).
        unique_bundles = np.unique(edges_net[:, BUNDLE_COL])

        # Pick cut distance based on the current number of FCNs (v1/v2) or
        # sub-bundles (v3) in this panel.
        # (Previously hardcoded to 5, which could disagree with current recut.)
        num_cuts = len(unique_bundles) if per_edge_leaves else num_fcns
        cut_distance = find_nth_largest_link(Z, num_cuts)
        if cut_distance is None:
            cut_distance = float(np.max(Z[:, 2])) if Z is not None and len(Z) > 0 else 0.0

        # Map bundle → FCN
        bundle_to_fcn = {}
        for b in unique_bundles:
            fcn_ids = edges_net[edges_net[:, BUNDLE_COL] == b, NETWORK_COL]
            fcn = int(fcn_ids[0]) if len(fcn_ids) > 0 else -1
            bundle_to_fcn[int(b)] = fcn

        # Map FCNs -> colors using the same logic as live plotting:
        # bundle override -> FCN "All" override -> default FCN color
        unique_fcns = sorted(set(bundle_to_fcn.values()))

        truncate_kwargs = {}
        if per_edge_leaves:
            # v3: Z has one leaf per EDGE, not per bundle -- label each leaf
            # by the sub-bundle *that specific edge* was cut into, and
            # truncate the display (scipy collapses deep subtrees into
            # count-labeled nodes) since a parent bundle can have tens of
            # thousands of edges/leaves, far too many to render individually.
            bundle_col_values = edges_net[:, BUNDLE_COL].astype(int)
            fcn_col_values = edges_net[:, NETWORK_COL].astype(int)
            labels = [
                f"B{bundle_col_values[i]} (FCN{fcn_col_values[i]})"
                for i in range(edges_net.shape[0])
            ]
            truncate_kwargs = {
                "truncate_mode": "lastp",
                "p": max(2 * len(unique_bundles), 10),
            }
        else:
            # v1/v2: Z has one leaf per bundle.
            labels = [
                f"B{int(b)} (FCN{bundle_to_fcn[int(b)]})"
                for b in unique_bundles
            ]

        # Build bundle_to_color → default FCN colors
        bundle_to_color = {}
        for b in unique_bundles:
            b_int = int(b)
            fcn = bundle_to_fcn.get(b_int, -1)

            bundle_color_idx = self.plotter.bundle_colors.get((fcn, b_int), None)
            if bundle_color_idx is None:
                bundle_color_idx = self.plotter.bundle_colors.get((fcn, "All"), None)

            if isinstance(bundle_color_idx, int):
                color_arr = my_colormap.colors[bundle_color_idx]
            elif fcn >= 0:
                color_arr = my_colormap.colors[fcn % len(my_colormap.colors)]
            else:
                color_arr = (0.5, 0.5, 0.5, 1.0)

            color_tuple = (
                tuple(color_arr.tolist())
                if hasattr(color_arr, "tolist")
                else tuple(color_arr)
            )
            bundle_to_color[b_int] = color_tuple

        # Build fcn_to_color
        fcn_to_color = {}
        for fcn in unique_fcns:
            # Check for custom FCN "All" color override
            fcn_color_idx = self.plotter.bundle_colors.get((fcn, "All"), None)

            if fcn_color_idx is not None:
                color_arr = my_colormap.colors[fcn_color_idx]
            else:
                color_arr = my_colormap.colors[fcn % len(my_colormap.colors)]

            color_tuple = (
                tuple(color_arr.tolist())
                if hasattr(color_arr, "tolist")
                else tuple(color_arr)
            )

            fcn_to_color[fcn] = color_tuple

        return {
            "Z": Z,
            "title": title,
            "labels": labels,
            "cut_distance": cut_distance,
            "fcn_to_color": fcn_to_color,
            "bundle_to_color": bundle_to_color,
            "unique_bundles": unique_bundles,
            "truncate_kwargs": truncate_kwargs,
        }

    def prepare_dendrogram_plot_data(self):
        '''
        Returns a list of panel dicts, one per dendrogram to display. v1/v2
        always produce exactly one panel (a single tree spanning every
        bundle). v3 produces one panel PER PARENT bundle, since each parent
        bundle has its own independent edge-linkage tree (see
        coffee_dac_pipeline_v3.py) -- these are never combined into one
        tree, so they are never drawn as one either.
        '''
        import numpy as np

        edges_net = self.edges_net
        per_edge_leaves = getattr(self, 'dendrogram_leaves', 'bundles') == 'edges'

        if not per_edge_leaves:
            return [
                self._prepare_dendrogram_panel(
                    edges_net, self.linkage_matrix, False, "FCN Dendrogram"
                )
            ]

        linkage_matrices = self.linkage_matrices or {}
        panels = []
        for parent_id in sorted(np.unique(edges_net[:, NETWORK_COL]).astype(int).tolist()):
            parent_edges = edges_net[edges_net[:, NETWORK_COL] == parent_id]
            Z = linkage_matrices.get(parent_id)
            if Z is None or len(Z) == 0:
                continue
            panels.append(self._prepare_dendrogram_panel(
                parent_edges, Z, True,
                f"Sub-bundle Dendrogram (v3 divisive) — parent bundle {parent_id}",
            ))
        return panels


    def show_dendrogram(self):
        if self.edges_net is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        panels = self.prepare_dendrogram_plot_data()
        if not panels:
            QMessageBox.information(
                self, "No Dendrogram",
                "No linkage tree is available to plot (nothing to link).",
            )
            return
        # v1/v2 always produce one panel; v3 produces one PER PARENT bundle,
        # since each parent bundle has its own independent tree that must
        # never be drawn merged with another parent's -- each panel opens
        # its own figure window.
        for panel in panels:
            show_dendrogram(
                Z=panel["Z"],
                labels=panel["labels"],
                cut_distance=panel["cut_distance"],
                fcn_to_color=panel["fcn_to_color"],
                bundle_to_color=panel["bundle_to_color"],
                title=panel["title"],
                **panel["truncate_kwargs"],
            )
 
