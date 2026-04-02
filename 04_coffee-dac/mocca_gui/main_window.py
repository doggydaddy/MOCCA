# mocca_gui/main_window.py
# mocca_gui/main_window.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QPushButton,
    QHBoxLayout, QLabel, QCheckBox, QSlider, QFileDialog, QMessageBox,
    QProgressDialog, QDialog, QVBoxLayout, QComboBox, QDialogButtonBox,
    QColorDialog, QSpinBox
)
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor

from mocca_gui.plotter import NetworkPlotter
from mocca_gui.tree_manager import TreeManager
from mocca_gui.fine_tuner import FineTuner
from mocca_gui.data_loader import EdgeDataLoader
from mocca_gui.gif_exporter import GifExporter
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

        splitter.setSizes([700, 300])

        self.preview_timer = None
        self.preview_frame = 0
    
    def load_data_dialog(self):
        from mocca_gui.data_loader import EdgeDataLoaderWorker
        from coffee_dac_pipeline import cache_exists
        from coffee_dac_pipeline_v2 import cache_exists_v2

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Edge CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        has_v1 = cache_exists(path)
        has_v2 = cache_exists_v2(path)

        # --- Build the prompt dialog when any cache exists ---
        pipeline = 'v1'
        use_cache = False
        recut = None

        if has_v1 or has_v2:
            dialog = QDialog(self)
            dialog.setWindowTitle("Cached results found")
            layout = QVBoxLayout(dialog)

            cache_info = []
            if has_v2:
                cache_info.append("  • v2 cache (processed CSV + linkage matrix)")
            if has_v1:
                cache_info.append("  • v1 cache (processed CSV + linkage matrix)")
            layout.addWidget(QLabel(
                "Previously processed results were found for this dataset:\n" +
                "\n".join(cache_info) +
                "\n\nHow would you like to proceed?"
            ))

            combo = QComboBox(dialog)
            if has_v2:
                combo.addItem("Load existing v2 results (fast)", ("v2", True))
            if has_v1:
                combo.addItem("Load existing v1 results (fast)", ("v1", True))
            combo.addItem("Re-process with pipeline v2 (slow)", ("v2", False))
            combo.addItem("Re-process with pipeline v1 (slow)", ("v1", False))
            layout.addWidget(combo)

            # Recut spinbox — only relevant when loading v2 cache
            recut_widget = QWidget(dialog)
            recut_layout = QHBoxLayout(recut_widget)
            recut_layout.setContentsMargins(0, 0, 0, 0)
            recut_label = QLabel("Cut into N networks (v2 cache only):")
            recut_spin = QSpinBox(dialog)
            recut_spin.setRange(2, 50)
            recut_spin.setValue(5)
            recut_layout.addWidget(recut_label)
            recut_layout.addWidget(recut_spin)
            layout.addWidget(recut_widget)

            def update_recut_visibility():
                choice_pipeline, choice_cache = combo.currentData()
                recut_widget.setVisible(choice_pipeline == 'v2' and choice_cache)
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
        self.linkage_matrix = result.get('linkage_matrix')
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
                plotter=self.plotter
            )
            QMessageBox.information(self, "Done", f"GIF saved:\n{filename}")

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

    def prepare_dendrogram_plot_data(self):
        import numpy as np
        from mocca_gui.colormap import my_colormap

        edges_net = self.edges_net
        Z = self.linkage_matrix

        # get number of FCNs
        num_fcns = int(np.max(edges_net[:, NETWORK_COL])) + 1

        def find_nth_largest_link(Z, n):
            # Z[:, 2] contains the distances of the merges
            distances = Z[:, 2]
            sorted_distances = np.sort(distances)[::-1]  # descending order
            if n > len(sorted_distances):
                return None
            nth_distance = sorted_distances[n - 1]
            return nth_distance
        
        # Find the 5th largest link distance
        cut_distance = find_nth_largest_link(Z, 5)

        # Get unique bundles
        unique_bundles = np.unique(edges_net[:, BUNDLE_COL])

        # Map bundle → FCN
        bundle_to_fcn = {}
        for b in unique_bundles:
            fcn_ids = edges_net[edges_net[:, BUNDLE_COL] == b, NETWORK_COL]
            fcn = int(fcn_ids[0]) if len(fcn_ids) > 0 else -1
            bundle_to_fcn[int(b)] = fcn

        # Map FCNs → colors from my_colormap
        unique_fcns = sorted(set(bundle_to_fcn.values()))
        fcn_colors = {}

        for i, fcn in enumerate(unique_fcns):
            color_arr = my_colormap.colors[i % len(my_colormap.colors)]
            color_tuple = (
                tuple(color_arr.tolist())
                if hasattr(color_arr, "tolist")
                else tuple(color_arr)
            )
            fcn_colors[fcn] = color_tuple

        # Build labels
        labels = [
            f"B{int(b)} (FCN{bundle_to_fcn[int(b)]})"
            for b in unique_bundles
        ]

        # Build bundle_to_color → default FCN colors
        bundle_to_color = {}
        for b in unique_bundles:
            b_int = int(b)
            fcn = bundle_to_fcn.get(b_int, -1)

            # Check for custom bundle color
            bundle_color_idx = self.plotter.bundle_colors.get((fcn, b_int), None)

            if isinstance(bundle_color_idx, int):
                # User picked custom color index → convert to tuple
                color_arr = my_colormap.colors[bundle_color_idx]
                color_tuple = (
                    tuple(color_arr.tolist())
                    if hasattr(color_arr, "tolist")
                    else tuple(color_arr)
                )
                bundle_to_color[b_int] = color_tuple
            else:
                # Default FCN color
                bundle_to_color[b_int] = fcn_colors.get(fcn, (0.5, 0.5, 0.5, 1.0))

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
            "labels": labels,
            "cut_distance": cut_distance,
            "fcn_to_color": fcn_to_color,
            "bundle_to_color": bundle_to_color,
            "unique_bundles": unique_bundles,
        }


    def show_dendrogram(self):
        if self.edges_net is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        prepared_data = self.prepare_dendrogram_plot_data()
        show_dendrogram(
            Z=self.linkage_matrix,
            labels=prepared_data["labels"],
            cut_distance=prepared_data["cut_distance"],
            fcn_to_color=prepared_data["fcn_to_color"],
            bundle_to_color=prepared_data["bundle_to_color"],
            title="FCN Dendrogram"
        )
 
