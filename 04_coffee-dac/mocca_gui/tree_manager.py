# mocca_gui/tree_manager.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QPushButton, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import Qt

class TreeManager:
    def __init__(self, main_window):
        self.main_window = main_window

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["FCNs and Bundles", "Selected Bundles"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        layout.addWidget(self.tree)

        self.bundle_color_buttons = {}  # Store color buttons for bundles

    def populate(self, edges_net):
        self.tree.clear()
        fcn_ids = sorted(set(edges_net[:, 7].astype(int)))

        for fcn in fcn_ids:
            fcn_item = QTreeWidgetItem([f"FCN {fcn}", ""])  # Right column will be filled later
            self.tree.addTopLevelItem(fcn_item)

            # Track selection summary for this FCN
            selection_summary = []

            # "All" item
            all_item = QTreeWidgetItem(["All"])
            all_item.setData(0, Qt.UserRole, {
                'type': 'all',
                'fcn': fcn,
                'bundle': 'All'
            })
            fcn_item.addChild(all_item)

            # Set buttons in column 1 for "All"
            all_buttons_widget = QWidget()
            all_buttons_layout = QHBoxLayout(all_buttons_widget)
            all_buttons_layout.setContentsMargins(0, 0, 0, 0)

            # Color button
            color_btn = QPushButton("Color")
            color_btn.setFixedSize(45, 20)
            color_btn.setToolTip(f"Set color for all bundles in FCN {fcn}")
            color_btn.clicked.connect(
                lambda _, f=fcn, btn=color_btn:
                    self.main_window.choose_fcn_color(f, btn)
            )
            all_buttons_layout.addWidget(color_btn)
            self.bundle_color_buttons[(fcn, "All")] = color_btn

            # Create the reset button
            reset_btn = QPushButton("↺")
            reset_btn.setFixedSize(30, 20)
            reset_btn.setToolTip("Reset color to FCN default")
            # Wire it to reset the bundle color
            reset_btn.clicked.connect(
                lambda _, f=fcn:
                    self.main_window.reset_fcn_color(f)
            )
            all_buttons_layout.addWidget(reset_btn)

            # Centroid toggle
            all_centroid_btn = QPushButton("Toggle All Centroids")
            all_centroid_btn.setFixedSize(150, 20)
            all_centroid_btn.setToolTip(f"Toggle all centroids in FCN {fcn}")
            all_centroid_btn.clicked.connect(
                lambda _, f=fcn:
                    self.main_window.toggle_all_centroids(f)
            )
            all_buttons_layout.addWidget(all_centroid_btn)

            # Fine-tune
            fine_tune_btn = QPushButton("Fine Tune")
            fine_tune_btn.setFixedSize(80, 20)
            fine_tune_btn.setToolTip(f"Fine tune FCN {fcn} (all bundles)")
            fine_tune_btn.clicked.connect(
                lambda _, f=fcn:
                    self.main_window.fine_tune_fcn(f)
            )
            all_buttons_layout.addWidget(fine_tune_btn)

            self.tree.setItemWidget(all_item, 1, all_buttons_widget)

            # Set background color if defined
            idx = self.main_window.plotter.bundle_colors.get((fcn, 'All'))
            if idx is not None:
                rgba = self.main_window.plotter.color_options[idx]
                color_btn.setStyleSheet(
                    f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]});"
                )
            else:
                color_btn.setStyleSheet("background-color: none; border: 1px solid #ccc;")

            # Add bundle items
            bundle_ids = sorted(set(edges_net[edges_net[:, 7] == fcn][:, 6].astype(int)))
            for bundle in bundle_ids:
                bundle_item = QTreeWidgetItem([f"Bundle {bundle}"])
                bundle_item.setData(0, Qt.UserRole, {
                    'type': 'bundle',
                    'fcn': fcn,
                    'bundle': bundle,
                    'bundle_id': str(bundle)
                })
                fcn_item.addChild(bundle_item)

                # Buttons for bundle
                bundle_buttons_widget = QWidget()
                bundle_buttons_layout = QHBoxLayout(bundle_buttons_widget)
                bundle_buttons_layout.setContentsMargins(0, 0, 0, 0)

                # color button
                color_btn = QPushButton("Color")
                color_btn.setFixedSize(45, 20)
                color_btn.setToolTip(f"Set color for Bundle {bundle}")
                color_btn.clicked.connect(
                    lambda _, f=fcn, b=bundle, btn=color_btn:
                        self.main_window.choose_color(f, b, btn)
                )
                bundle_buttons_layout.addWidget(color_btn)
                self.bundle_color_buttons[(fcn, bundle)] = color_btn

                idx = self.main_window.plotter.bundle_colors.get((fcn, bundle))
                if idx is not None:
                    rgba = self.main_window.plotter.color_options[idx]
                    color_btn.setStyleSheet(
                        f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]});"
                    )
                else:
                    color_btn.setStyleSheet("background-color: none; border: 1px solid #ccc;")

                # Create the reset button
                reset_btn = QPushButton("↺")
                reset_btn.setFixedSize(30, 20)
                reset_btn.setToolTip("Reset color to FCN default")
                # Wire it to reset the bundle color
                reset_btn.clicked.connect(
                    lambda _, f=fcn, b=bundle: self.main_window.reset_bundle_color(f, b)
                )
                bundle_buttons_layout.addWidget(reset_btn)

                # Centroid toggle 
                centroid_btn = QPushButton("Centroid")
                centroid_btn.setFixedSize(80, 20)
                centroid_btn.setToolTip(f"Toggle centroid view for Bundle {bundle}")
                centroid_btn.clicked.connect(
                    lambda _, f=fcn, b=bundle, btn=centroid_btn:
                        self.main_window.toggle_centroid(f, b, btn)
                )
                bundle_buttons_layout.addWidget(centroid_btn)

                # Fine-tune button
                fine_tune_btn = QPushButton("Fine Tune")
                fine_tune_btn.setFixedSize(80, 20)
                fine_tune_btn.setToolTip(f"Fine tune FCN {fcn}, Bundle {bundle}")
                fine_tune_btn.clicked.connect(
                    lambda _, f=fcn, b=bundle:
                        self.main_window.fine_tune_bundle(f, b)
                )
                bundle_buttons_layout.addWidget(fine_tune_btn)

                self.tree.setItemWidget(bundle_item, 1, bundle_buttons_widget)

            # ✅ OPTIONAL: Clear "Selected Bundles" label unless you want to pre-fill it
            fcn_item.setText(1, "")  # You can dynamically update this during selection

            self.tree.itemSelectionChanged.connect(self.update_selected_bundles_column)
    
    def update_selected_bundles_column(self):
        # Clear previous text
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setText(1, "")

        selected = self.tree.selectedItems()
        fcn_to_bundles = {}

        for item in selected:
            parent = item.parent()
            text = item.text(0)

            if parent:  # Bundle or All
                fcn_text = parent.text(0)
                fcn = int(fcn_text.split()[1])
                if fcn not in fcn_to_bundles:
                    fcn_to_bundles[fcn] = []
                if text == "All":
                    fcn_to_bundles[fcn] = ["All"]
                elif text.startswith("Bundle "):
                    if "All" not in fcn_to_bundles[fcn]:  # Don't mix with 'All'
                        bundle = int(text.split()[1])
                        fcn_to_bundles[fcn].append(str(bundle))

            else:  # Top-level FCN
                if text.startswith("FCN "):
                    fcn = int(text.split()[1])
                    fcn_to_bundles[fcn] = ["All"]

        # Write summary text into column 1 for each FCN
        for i in range(self.tree.topLevelItemCount()):
            fcn_item = self.tree.topLevelItem(i)
            fcn = int(fcn_item.text(0).split()[1])
            bundles = fcn_to_bundles.get(fcn)
            if bundles:
                if "All" in bundles:
                    fcn_item.setText(1, "All")
                else:
                    fcn_item.setText(1, ", ".join(bundles))

    def get_selection(self):
        selected_items = self.tree.selectedItems()
        selection = []

        for item in selected_items:
            text = item.text(0)
            parent = item.parent()

            if parent:  # Bundle or 'All' inside FCN
                fcn_text = parent.text(0)
                fcn = int(fcn_text.split()[1])
                if text == "All":
                    selection.append({"fcn": fcn, "bundle": "All"})
                elif text.startswith("Bundle "):
                    bundle = int(text.split()[1])
                    selection.append({"fcn": fcn, "bundle": bundle})
            else:  # Top-level FCN node
                if text.startswith("FCN "):
                    fcn = int(text.split()[1])
                    selection.append({"fcn": fcn, "bundle": "All"})

        return selection

    def clear_selection(self):
        for item in self.tree.selectedItems():
            item.setSelected(False)
        self.tree.clearSelection()