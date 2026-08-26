# imports

# general
import numpy as np
import os

# data loading
import pandas as pd
import nibabel as nib

# clustering libraries
import scipy.cluster.hierarchy
import sklearn.cluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# distance eval
import matplotlib.pyplot as plt

# visualization
import pyvista as pv
import matplotlib as mpl



# Define the colors we want to use, currently supports up to 13 * 2 = 26 colours
# For more colors, https://www.rapidtables.com/web/color/RGB_Color.html

from matplotlib.colors import ListedColormap

one = np.array([153/256, 0.0, 0.0, 1.0])
two = np.array([153/256, 76/256, 0.0, 1.0])
three = np.array([153/256, 153/256, 0.0, 1.0])
four = np.array([76/256, 153/256, 0.0, 1.0])
five = np.array([0/256, 153/256, 0.0, 1.0])
six = np.array([0/256, 153/256, 76/256, 1.0])
seven = np.array([0/256, 153/256, 153/256, 1.0])
eight = np.array([0/256, 76/256, 153/256, 1.0])
nine = np.array([0/256, 0/256, 153/256, 1.0])
ten = np.array([76/256, 0/256, 153/256, 1.0])
eleven = np.array([153/256, 0/256, 153/256, 1.0])
twelve = np.array([153/256, 0/256, 76/256, 1.0])
thirteen = np.array([64/256, 64/256, 164/256, 1.0])

fourteen = np.array([255/256, 102/256, 102/256, 1.0])
fifteen = np.array([255/256, 178/256, 102/256, 1.0])
sixteen = np.array([255/256, 255/256, 102/256, 1.0])
seventeen = np.array([178/256, 255/256, 102/256, 1.0])
eighteen = np.array([102/256, 255/256, 102/256, 1.0])
nineteen = np.array([102/256, 255/256, 178/256, 1.0])
twenty = np.array([102/256, 255/256, 255/256, 1.0])
twentyone = np.array([102/256, 178/256, 255/256, 1.0])
twentytwo = np.array([102/256, 102/256, 255/256, 1.0])
twentythree = np.array([178/256, 102/256, 255/256, 1.0])
twentyfour = np.array([255/256, 102/256, 255/256, 1.0])
twentyfive = np.array([255/256, 102/256, 178/256, 1.0])
twentysix = np.array([192/256, 192/256, 192/256, 1.0])

twentyseven = np.array([255/256, 0/256, 0/256, 1.0])
twentyeight = np.array([255/256, 128/256, 0/256, 1.0])
twentynine = np.array([255/256, 255/256, 0/256, 1.0])
thirty = np.array([128/256, 255/256, 0/256, 1.0])
thirtyone = np.array([0/256, 255/256, 0/256, 1.0])
thirtytwo = np.array([0/256, 255/256, 128/256, 1.0])
thirtythree = np.array([0/256, 255/256, 255/256, 1.0])
thirtyfour = np.array([0/256, 128/256, 255/256, 1.0])
thirtyfive = np.array([0/256, 0/256, 255/256, 1.0])
thirtysix = np.array([128/256, 0/256, 255/256, 1.0])
thirtyseven = np.array([255/256, 0/256, 255/256, 1.0])
thirtyeight = np.array([255/256, 0/256, 128/256, 1.0])
thirtynine = np.array([128/256, 128/256, 128/256, 1.0])



mapping = np.linspace(1, 14, 39)
newcolors = np.empty((39, 4))
newcolors[0] = one
newcolors[1] = two
newcolors[2] = three
newcolors[3] = four
newcolors[4] = five
newcolors[5] = six
newcolors[6] = seven
newcolors[7] = eight
newcolors[8] = nine
newcolors[9] = ten
newcolors[10] = eleven
newcolors[11] = twelve
newcolors[12] = thirteen
newcolors[13] = fourteen
newcolors[14] = fifteen
newcolors[15] = sixteen
newcolors[16] = seventeen
newcolors[17] = eighteen
newcolors[18] = nineteen
newcolors[19] = twenty
newcolors[20] = twentyone
newcolors[21] = twentytwo
newcolors[22] = twentythree
newcolors[23] = twentyfour
newcolors[24] = twentyfive
newcolors[25] = twentysix
newcolors[26] = twentyseven
newcolors[27] = twentyeight
newcolors[28] = twentynine
newcolors[29] = thirty
newcolors[30] = thirtyone
newcolors[31] = thirtytwo
newcolors[32] = thirtythree
newcolors[33] = thirtyfour
newcolors[34] = thirtyfive
newcolors[35] = thirtysix
newcolors[36] = thirtyseven
newcolors[37] = thirtyeight
newcolors[38] = thirtynine

# Make the colormap from the listed colors
my_colormap = ListedColormap(newcolors)

    
def plotline_ijk(plotter, edge, plot_endpoints=True, current_eps):
    '''
    Plots a line and its endpoints if selected.
    '''
    
    # define opacity values for edge and (if relevant) the end-points.
    endpoint_opacity = 0.2
    edge_opacity = 0.8
    this_color = my_colormap(int(edge[7]))

    # endpoints for the spline
    a = edge[0:3]
    b = edge[3:6]
    
    # constructing midpoint with offset as "midpoint" to make the splines a bit nicer.
    offset_i = np.abs(a[0] - b[0]) * 0.1
    offset_j = np.abs(a[1] - b[1]) * 0.1
    offset_k = np.abs(a[2] - b[2]) * 0.1
    m = [((a[0]+b[0])/2)+offset_i, 
         ((a[1]+b[1])/2)+offset_j, 
         ((a[2]+b[2])/2)+offset_k]
    
    # defining the spline
    spline_points = np.row_stack((a, m, b))
    
    N = 32 # how many spline segments
    spline = pv.Spline( spline_points, N )
    spline['scalars'] = np.full(N, edge[6])
    
    # plot the edge
    plotter.add_mesh(spline, color=this_color, render_lines_as_tubes=True, line_width=3, opacity=edge_opacity, show_scalar_bar=True)
    
    # plot end-points if plot_endpoints flag is True
    if plot_endpoints:
        d=np.sqrt(3)/4
        a_box = pv.Box([edge[0]-d, edge[0]+d, edge[1]-d, edge[1]+d, edge[2]-d, edge[2]+d])
        b_box = pv.Box([edge[3]-d, edge[3]+d, edge[4]-d, edge[4]+d, edge[5]-d, edge[5]+d])
        plotter.add_mesh(a_box, color=this_color, opacity=endpoint_opacity)
        plotter.add_mesh(b_box, color=this_color, opacity=endpoint_opacity)
        
    



# load edges with network clusterings
input_csv = 'intermediate/edges_edclust.csv'

filename, extension = os.path.splitext(input_csv)

edges_net_csv = pd.read_csv(input_csv)
edges_net = edges_net_csv.to_numpy()


# output each network to gif

# load background now so we do not have to do it in the loop
reader = pv.get_reader("intermediate/test.stl")
mesh = reader.read()


p = pv.Plotter(notebook=False, off_screen=False)

# add brain stl background
p.add_mesh(mesh, show_edges=True, edge_color="gray", color='white', point_size=3, render_points_as_spheres=True, opacity=0.1, show_scalar_bar=True)
      
# plot individual network
network_id = 0
edges_to_plot = edges_net[edges_net[:, 7] == network_id, :]
nr_edges = edges_to_plot.shape[0]
for j in range(nr_edges):
    plotline_ijk(p, edges_to_plot[j, :])
p.show()