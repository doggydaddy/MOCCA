# mocca_gui/plotter.py

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from PyQt5.QtWidgets import QApplication

from mocca_gui.colormap import my_colormap

def plotline_ijk(plotter, edge, color, offset_multiplier=1.0, line_width=3, opacity=0.8):
    a = edge[0:3]
    b = edge[3:6]
    offset = (np.abs(a - b) * 0.1 * offset_multiplier)
    m = ((a + b) / 2) + offset
    spline_points = np.vstack((a, m, b))
    spline = pv.Spline(spline_points, 32)
    plotter.add_mesh(
        spline,
        color=color[:3],
        line_width=line_width,
        render_lines_as_tubes=True,
        opacity=opacity
    )

def add_endpoints(plotter, edge, color, size_scale=1.0, opacity=0.2):
    a = edge[0:3]
    b = edge[3:6]
    d = np.sqrt(3)/4 * size_scale

    box_a = pv.Box([
        a[0] - d, a[0] + d,
        a[1] - d, a[1] + d,
        a[2] - d, a[2] + d
    ])
    box_b = pv.Box([
        b[0] - d, b[0] + d,
        b[1] - d, b[1] + d,
        b[2] - d, b[2] + d
    ])

    plotter.add_mesh(box_a, color=color[:3], opacity=opacity)
    plotter.add_mesh(box_b, color=color[:3], opacity=opacity)

def generate_centroid_edge(edges_bundle, plotter=None, color=None):
    start_points = edges_bundle[:, 0:3]
    end_points = edges_bundle[:, 3:6]

    centroid_start = np.mean(start_points, axis=0)
    centroid_end = np.mean(end_points, axis=0)

    cluster_id = edges_bundle[0, 6]
    network_id = edges_bundle[0, 7]

    centroid_edge = np.concatenate([
        centroid_start,
        centroid_end,
        np.array([cluster_id, network_id]),
    ])

    boxes = []
    if plotter is not None:
        d = np.sqrt(3)/4 * 1.0
        for point in start_points:
            box = pv.Box([
                point[0] - d, point[0] + d,
                point[1] - d, point[1] + d,
                point[2] - d, point[2] + d
            ])
            plotter.add_mesh(box, color=color[:3], opacity=0.2)
            boxes.append(box)

        for point in end_points:
            box = pv.Box([
                point[0] - d, point[0] + d,
                point[1] - d, point[1] + d,
                point[2] - d, point[2] + d
            ])
            plotter.add_mesh(box, color=color[:3], opacity=0.2)
            boxes.append(box)

    return centroid_edge, boxes

class NetworkPlotter:
    def __init__(self, plotter, brain_mesh_path):
        self.plotter = plotter
        self.brain_mesh_path = brain_mesh_path
        self.brain_mesh = pv.read(brain_mesh_path)

        self.thicknesses = {}
        self.curvatures = {}
        self.endpoint_sizes = {}

        self.bundle_colors = {}  # Key: (fcn, bundle) → color index
        self.centroid_flags = {}  # (fcn, bundle) → True/False
        self.opacities = {}  # (fcn, bundle) → float


    def clear(self):
        self.plotter.clear()
        self.plotter.add_mesh(self.brain_mesh, opacity=0.1, color='grey')

    def draw_selection(
        self,
        edges_net,
        selection,
        endpoint_visible=True,
        stop_flag=None,
        progress_callback=None
    ):
        self.clear()

        total_edges = sum(
            len(edges_net[
                (edges_net[:,7] == item['fcn']) &
                (edges_net[:,6] == b)
            ])
            for item in selection
            for b in (
                np.unique(edges_net[edges_net[:,7]==item['fcn']][:,6])
                if item['bundle'] == "All"
                else [item['bundle']]
            )
        )
        if total_edges == 0:
            total_edges = 1

        edges_drawn = 0

        for item in selection:
            fcn = item['fcn']
            bundle = item['bundle']

            if bundle == "All":
                bundles = np.unique(edges_net[edges_net[:,7]==fcn][:,6])
            else:
                bundles = [bundle]

            for b in bundles:
                # check bundle color first, then FCN-wide color
                idx = self.bundle_colors.get((fcn, int(b)))

                if idx is None:
                    idx = self.bundle_colors.get((fcn, 'All'))

                if idx is not None:
                    color = my_colormap.colors[idx]
                else:
                    color = my_colormap.colors[fcn % len(my_colormap.colors)]
                    #color_list = np.array([
                #"""more contrasted FCN-colors:"""" color = np.array([[153/255, 0.0, 0.0, 1.0], [0.0, 153/255, 0.0, 1.0], [0.0, 0.0, 153/255, 1.0], [255/255, 128/255, 0.0, 1.0],[128/255, 0.0, 255/255, 1.0]])
                
                # Get centroid toggle state
                use_centroid = self.centroid_flags.get((fcn, int(b)), False)

                edges = edges_net[
                    (edges_net[:,7] == fcn) &
                    (edges_net[:,6] == b)
                ]

                if use_centroid and len(edges) > 0:
                    # generate centroid edge
                    centroid_edge, boxes = generate_centroid_edge(
                        edges,
                        plotter =self.plotter,
                        color=color
                    )
                
                    plotline_ijk(
                        self.plotter,
                        centroid_edge,
                        color=color,
                        offset_multiplier=self.curvatures.get((fcn, int(b)), 1.0),
                        line_width=self.thicknesses.get((fcn, int(b)), 3),
                        opacity=self.opacities.get((fcn, int(b)), 0.8)
                    )

                else: 

                    for edge in edges:
                        if stop_flag and stop_flag():
                            print("Plotting cancelled.")
                            return

                        thickness = self.thicknesses.get((fcn, int(b)), 3)
                        curvature = self.curvatures.get((fcn, int(b)), 1.0)
                        endpoint_size = self.endpoint_sizes.get((fcn, int(b)), 1.0)

                        plotline_ijk(
                            self.plotter,
                            edge,
                            color=color,
                            offset_multiplier=curvature,
                            line_width=thickness,
                            opacity=self.opacities.get((fcn, int(b)), 0.8)
                        )

                        if endpoint_visible:
                            add_endpoints(
                                self.plotter,
                                edge,
                                color=color,
                                size_scale=endpoint_size,
                                opacity=self.opacities.get((fcn, int(b)), 0.8)
                            )

                        edges_drawn += 1

                        # call processEvents() every N edges
                        if edges_drawn % 10 == 0:
                            QApplication.processEvents()

                            if progress_callback:
                                percent = int((edges_drawn / total_edges) * 100)
                                progress_callback(percent)

        self.plotter.reset_camera()
        self.plotter.render()


