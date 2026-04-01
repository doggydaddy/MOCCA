# mocca_gui/fine_tuner.py

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QDialog, QDialogButtonBox, QComboBox
)
from PyQt5.QtCore import Qt

class FineTuner:
    def __init__(self, main_window):
        self.main_window = main_window

        self.widget = QWidget()
        layout = QHBoxLayout(self.widget)

        self.label = QLabel("Fine Tuning: (none)")
        layout.addWidget(self.label)

        self.btn_thickness = QPushButton("Thickness: -")
        self.btn_thickness.setFixedHeight(20)
        self.btn_thickness.setEnabled(False)
        self.btn_thickness.clicked.connect(self.set_thickness)
        layout.addWidget(self.btn_thickness)

        self.btn_curvature = QPushButton("Curvature: -")
        self.btn_curvature.setFixedHeight(20)
        self.btn_curvature.setEnabled(False)
        self.btn_curvature.clicked.connect(self.set_curvature)
        layout.addWidget(self.btn_curvature)

        self.btn_endpoint = QPushButton("Endpoints: -")
        self.btn_endpoint.setFixedHeight(20)
        self.btn_endpoint.setEnabled(False)
        self.btn_endpoint.clicked.connect(self.set_endpoint_size)
        layout.addWidget(self.btn_endpoint)

        self.btn_opacity = QPushButton("Opacity: -")
        self.btn_opacity.setFixedHeight(20)
        self.btn_opacity.setEnabled(False)
        self.btn_opacity.clicked.connect(self.set_opacity)
        layout.addWidget(self.btn_opacity)

        self.reset()

    def reset(self):
        self.mode = None
        self.fcn = None
        self.bundle = None
        self.btn_thickness.setEnabled(False)
        self.btn_curvature.setEnabled(False)
        self.btn_endpoint.setEnabled(False)
        self.btn_opacity.setEnabled(False)
        self.label.setText("Fine Tuning: (none)")

    def load_target(self, mode, fcn=None, bundle=None):
        self.mode = mode
        self.fcn = fcn
        self.bundle = bundle

        if mode == "bundle":
            text = f"Fine Tuning: FCN {fcn}, Bundle {bundle}"
        elif mode == "fcn":
            text = f"Fine Tuning: FCN {fcn} (all bundles)"
        elif mode == "global":
            text = "Fine Tuning: All Networks"
        else:
            text = "Fine Tuning: (none)"

        self.label.setText(text)

        self.btn_thickness.setEnabled(True)
        self.btn_curvature.setEnabled(True)
        self.btn_endpoint.setEnabled(True)
        self.btn_opacity.setEnabled(True)

        # Load current values from plotter dictionaries
        mw = self.main_window

        thickness = self.get_current_thickness()
        curvature = self.get_current_curvature()
        endpoint_size = self.get_current_endpoint_size()
        opacity = self.get_current_opacity()

        self.btn_thickness.setText(f"Thickness: {thickness}")
        self.btn_curvature.setText(f"Curvature: {curvature:.2f}x")
        self.btn_endpoint.setText(f"Endpoints: {endpoint_size:.2f}x")
        self.btn_opacity.setText(f"Opacity: {opacity:.2f}")

    def get_current_thickness(self):
        mw = self.main_window
        default = 3

        if self.mode == "bundle":
            return mw.plotter.thicknesses.get((self.fcn, self.bundle), default)
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            if len(bundles) > 0:
                b = int(bundles[0])
                return mw.plotter.thicknesses.get((self.fcn, b), default)
            else:
                return default
        elif self.mode == "global":
            all_bundles = mw.edges_net[:,[6,7]]
            if len(all_bundles) > 0:
                f, b = int(all_bundles[0][1]), int(all_bundles[0][0])
                return mw.plotter.thicknesses.get((f, b), default)
            else:
                return default
        else:
            return default

    def get_current_curvature(self):
        mw = self.main_window
        default = 1.0

        if self.mode == "bundle":
            return mw.plotter.curvatures.get((self.fcn, self.bundle), default)
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            if len(bundles) > 0:
                b = int(bundles[0])
                return mw.plotter.curvatures.get((self.fcn, b), default)
            else:
                return default
        elif self.mode == "global":
            all_bundles = mw.edges_net[:,[6,7]]
            if len(all_bundles) > 0:
                f, b = int(all_bundles[0][1]), int(all_bundles[0][0])
                return mw.plotter.curvatures.get((f, b), default)
            else:
                return default
        else:
            return default

    def get_current_endpoint_size(self):
        mw = self.main_window
        default = 1.0

        if self.mode == "bundle":
            return mw.plotter.endpoint_sizes.get((self.fcn, self.bundle), default)
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            if len(bundles) > 0:
                b = int(bundles[0])
                return mw.plotter.endpoint_sizes.get((self.fcn, b), default)
            else:
                return default
        elif self.mode == "global":
            all_bundles = mw.edges_net[:,[6,7]]
            if len(all_bundles) > 0:
                f, b = int(all_bundles[0][1]), int(all_bundles[0][0])
                return mw.plotter.endpoint_sizes.get((f, b), default)
            else:
                return default
        else:
            return default

    def get_current_opacity(self):
        mw = self.main_window
        default = 0.8

        if self.mode == "bundle":
            return mw.plotter.opacities.get((self.fcn, self.bundle), default)
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            if len(bundles) > 0:
                b = int(bundles[0])
                return mw.plotter.opacities.get((self.fcn, b), default)
            else:
                return default
        elif self.mode == "global":
            all_bundles = mw.edges_net[:,[6,7]]
            if len(all_bundles) > 0:
                f, b = int(all_bundles[0][1]), int(all_bundles[0][0])
                return mw.plotter.opacities.get((f, b), default)
            else:
                return default
        else:
            return default

    def set_thickness(self):
        val = self._get_thickness_dialog()
        if val is None:
            return

        mw = self.main_window
        if self.mode == "bundle":
            mw.plotter.thicknesses[(self.fcn, self.bundle)] = val
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            for b in bundles:
                mw.plotter.thicknesses[(self.fcn, int(b))] = val
        elif self.mode == "global":
            for row in mw.edges_net:
                fcn = int(row[7])
                bundle = int(row[6])
                mw.plotter.thicknesses[(fcn, bundle)] = val

        self.btn_thickness.setText(f"Thickness: {val}")
        mw.plot_selected()

    def set_curvature(self):
        val = self._get_curvature_dialog()
        if val is None:
            return

        mw = self.main_window
        if self.mode == "bundle":
            mw.plotter.curvatures[(self.fcn, self.bundle)] = val
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            for b in bundles:
                mw.plotter.curvatures[(self.fcn, int(b))] = val
        elif self.mode == "global":
            for row in mw.edges_net:
                fcn = int(row[7])
                bundle = int(row[6])
                mw.plotter.curvatures[(fcn, bundle)] = val

        self.btn_curvature.setText(f"Curvature: {val:.2f}x")
        mw.plot_selected()

    def set_endpoint_size(self):
        val = self._get_endpoint_size_dialog()
        if val is None:
            return

        mw = self.main_window
        if self.mode == "bundle":
            mw.plotter.endpoint_sizes[(self.fcn, self.bundle)] = val
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            for b in bundles:
                mw.plotter.endpoint_sizes[(self.fcn, int(b))] = val
        elif self.mode == "global":
            for row in mw.edges_net:
                fcn = int(row[7])
                bundle = int(row[6])
                mw.plotter.endpoint_sizes[(fcn, bundle)] = val

        self.btn_endpoint.setText(f"Endpoints: {val:.2f}x")
        mw.plot_selected()

    def set_opacity(self):
        val = self._get_opacity_dialog()
        if val is None:
            return

        mw = self.main_window
        if self.mode == "bundle":
            mw.plotter.opacities[(self.fcn, self.bundle)] = val
        elif self.mode == "fcn":
            bundles = mw.edges_net[mw.edges_net[:,7] == self.fcn][:,6]
            for b in bundles:
                mw.plotter.opacities[(self.fcn, int(b))] = val
        elif self.mode == "global":
            for row in mw.edges_net:
                fcn = int(row[7])
                bundle = int(row[6])
                mw.plotter.opacities[(fcn, bundle)] = val

        self.btn_opacity.setText(f"Opacity: {val:.2f}")
        mw.plot_selected()

    def _get_thickness_dialog(self):
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Choose Line Thickness")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for val in [1, 2, 3, 4, 6, 8]:
            combo.addItem(f"{val} px", val)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentData()
        return None

    def _get_curvature_dialog(self):
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Choose Curvature Offset Multiplier")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for val in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            combo.addItem(f"{val:.2f}x", val)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentData()
        return None

    def _get_endpoint_size_dialog(self):
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Choose Endpoint Box Size Multiplier")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for val in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            combo.addItem(f"{val:.2f}x", val)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentData()
        return None

    def _get_opacity_dialog(self):
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Choose Opacity (Transparency)")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            combo.addItem(f"{val:.2f}", val)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentData()
        return None


