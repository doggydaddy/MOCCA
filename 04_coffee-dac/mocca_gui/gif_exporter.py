# mocca_gui/gif_exporter.py

import pyvista as pv
import os
from mocca_gui.plotter import NetworkPlotter

class GifExporter:
    def __init__(self):
        pass

    def export(self, edges_net, selection, filename, elevation, azimuth, plotter):
        import pyvista as pv
    
        p = pv.Plotter(off_screen=True)
        brain_mesh = pv.read("test.stl")
        p.add_mesh(brain_mesh, opacity=0.1, color='grey')
    
        # ✅ Create a NetworkPlotter for this off-screen plotter
        gif_plotter = NetworkPlotter(p, brain_mesh_path="test.stl")
    
        # ✅ Plot the same geometry as the GUI
        gif_plotter.draw_selection(
            edges_net,
            selection,
            endpoint_visible=True
        )
    
        p.open_gif(filename)
    
        for angle in range(0, 360, 2):
            p.camera.azimuth = azimuth + angle
            p.camera.elevation = elevation
            p.write_frame()
    
        p.close()
