# mocca_gui/gif_exporter.py

import pyvista as pv
import os
from mocca_gui.plotter import NetworkPlotter

class GifExporter:
    def __init__(self):
        pass

    def export(self, edges_net, selection, filename, elevation, azimuth,
               plotter, endpoint_visible=True):
        import pyvista as pv

        p = pv.Plotter(off_screen=True)

        # ✅ Create a NetworkPlotter for this off-screen plotter
        gif_plotter = NetworkPlotter(p)

        # ✅ Mirror all visualization state from the live plotter so the GIF
        #    matches exactly what is shown in the GUI (centroids, thickness,
        #    curvature, opacity, endpoint size, per-bundle colours, …)
        gif_plotter.centroid_flags   = dict(plotter.centroid_flags)
        gif_plotter.thicknesses      = dict(plotter.thicknesses)
        gif_plotter.curvatures       = dict(plotter.curvatures)
        gif_plotter.opacities        = dict(plotter.opacities)
        gif_plotter.endpoint_sizes   = dict(plotter.endpoint_sizes)
        gif_plotter.bundle_colors    = dict(plotter.bundle_colors)

        # ✅ Plot the same geometry as the GUI
        gif_plotter.draw_selection(
            edges_net,
            selection,
            endpoint_visible=endpoint_visible,
        )
    
        p.open_gif(filename)
    
        for angle in range(0, 360, 2):
            p.camera.azimuth = azimuth + angle
            p.camera.elevation = elevation
            p.write_frame()
    
        p.close()
