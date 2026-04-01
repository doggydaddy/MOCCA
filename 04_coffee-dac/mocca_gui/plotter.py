# mocca_gui/plotter.py

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from PyQt5.QtWidgets import QApplication

from mocca_gui.colormap import my_colormap
from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL

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
    points = pv.PolyData(np.vstack([a, b]))
    geom = pv.Box(bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    glyphs = points.glyph(geom=geom, scale=False, factor=np.sqrt(3)/4 * size_scale * 2)
    plotter.add_mesh(glyphs, color=color[:3], opacity=opacity)

def generate_centroid_edge(edges_bundle, plotter=None, color=None):
    start_points = edges_bundle[:, 0:3]
    end_points = edges_bundle[:, 3:6]

    centroid_start = np.mean(start_points, axis=0)
    centroid_end = np.mean(end_points, axis=0)

    cluster_id = edges_bundle[0, BUNDLE_COL]
    network_id = edges_bundle[0, NETWORK_COL]

    centroid_edge = np.concatenate([
        centroid_start,
        centroid_end,
        np.array([cluster_id, network_id]),
    ])

    boxes = []
    if plotter is not None:
        d = np.sqrt(3)/4 * 1.0
        all_points = np.vstack([start_points, end_points])
        cloud = pv.PolyData(all_points)
        geom = pv.Box(bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
        glyphs = cloud.glyph(geom=geom, scale=False, factor=d * 2)
        actor = plotter.add_mesh(glyphs, color=color[:3], opacity=0.2)
        boxes.append(actor)

    return centroid_edge, boxes

class NetworkPlotter:
    # Default brain mesh layers: (filename, base_opacity, color, smooth_shading)
    # base_opacity is the value at slider=100%; slider scales it linearly down to 0.
    # smooth_shading=True computes per-vertex normals so gyri/sulci show relief.
    DEFAULT_BRAIN_MESHES = [
        ("brain3mm_wm.stl",    0.30, "#8B7355", False),  # WM interior – flat ok
        ("brain3mm_gm.stl",    0.25, "#C0C0C0", True),   # GM cortex – smooth shading for gyri/sulci
        ("brain3mm_outer.stl", 0.10, "#D8D8D8", True),   # outer hull
        ("brain3mm.stl",       0.05, "#D8D8D8", False),  # silhouette only
    ]

    def __init__(self, plotter, brain_mesh_path=None, brain_meshes=None):
        """
        Parameters
        ----------
        plotter          : pyvista interactor
        brain_mesh_path  : str, legacy single-mesh path (kept for compatibility)
        brain_meshes     : list of (path, opacity, color) tuples.
                           If provided, overrides brain_mesh_path.
                           If neither is given, DEFAULT_BRAIN_MESHES is used.
        """
        self.plotter = plotter

        if brain_meshes is not None:
            self._brain_meshes = brain_meshes
        elif brain_mesh_path is not None:
            # legacy: single mesh at default opacity
            self._brain_meshes = [(brain_mesh_path, 0.25, "grey", True)]
        else:
            self._brain_meshes = self.DEFAULT_BRAIN_MESHES

        # Pre-load all brain meshes; compute smooth normals where requested
        self._brain_mesh_actors = []
        for entry in self._brain_meshes:
            path, _opacity, _color = entry[0], entry[1], entry[2]
            _smooth = entry[3] if len(entry) > 3 else False
            try:
                mesh = pv.read(path)
                if _smooth:
                    mesh = mesh.compute_normals(cell_normals=False, point_normals=True,
                                                split_vertices=False, consistent_normals=True)
                self._brain_mesh_actors.append((mesh, _opacity, _color, _smooth))
            except Exception as e:
                print(f"Warning: could not load brain mesh '{path}': {e}")

        self.thicknesses = {}
        self.curvatures = {}
        self.endpoint_sizes = {}

        self.bundle_colors = {}  # Key: (fcn, bundle) → color index
        self.centroid_flags = {}  # (fcn, bundle) → True/False
        self.opacities = {}  # (fcn, bundle) → float
        self.brain_opacity_scale = 0.5  # multiplier applied to all brain mesh opacities

        # Add brain meshes once and keep their actors for live opacity updates
        self._live_brain_actors = []
        self._add_brain_meshes()

    def _add_brain_meshes(self):
        """Add brain mesh layers to the plotter and store the returned actors."""
        self._live_brain_actors = []
        for mesh, base_opacity, color, smooth in self._brain_mesh_actors:
            actor = self.plotter.add_mesh(
                mesh,
                opacity=base_opacity * self.brain_opacity_scale,
                color=color,
                smooth_shading=smooth,
                lighting=True,
            )
            self._live_brain_actors.append((actor, base_opacity))

    def set_brain_opacity(self, scale):
        """Update brain mesh opacity in-place without redrawing edges."""
        self.brain_opacity_scale = max(0.0, min(1.0, scale))
        for actor, base_opacity in self._live_brain_actors:
            actor.GetProperty().SetOpacity(base_opacity * self.brain_opacity_scale)
        self.plotter.render()

    def clear(self):
        """Clear all actors instantly, then restore persistent brain meshes."""
        self.plotter.clear()
        self._add_brain_meshes()

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
                (edges_net[:,NETWORK_COL] == item['fcn']) &
                (edges_net[:,BUNDLE_COL] == b)
            ])
            for item in selection
            for b in (
                np.unique(edges_net[edges_net[:,NETWORK_COL]==item['fcn']][:,BUNDLE_COL])
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
                bundles = np.unique(edges_net[edges_net[:,NETWORK_COL]==fcn][:,BUNDLE_COL])
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
                    (edges_net[:,NETWORK_COL] == fcn) &
                    (edges_net[:,BUNDLE_COL] == b)
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


