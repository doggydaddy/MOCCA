# GUI for COFFEE-DAC

A Graphic User Interface for COFFEE-DAC, a part of the MOCCA pipeline for
analysis of resting-state fMRI data.

COFFEE-DAC-program provides brain connection data from more raw fMRI-dataset to
more intuitive 3D-graphs, where the connections are represented by bundles of
connections and these bundles are grouped into FCNs (Functional Connectivity
Networks).

# Dependencies

This GUI is dependent on  the COFFEE-DAC program, ijk-csv-data (see below)
processed from the MOCCA analysis pipeline and the *brain_template.stl* with the
specific brain mesh belonging to the COFFEE-DAC pipeline analysis program.

# Prequisites

## Required Packages

List of required packages in $requirements.txt$:

* pandas
* PyQt5
* pyvistaqt
* nibabel
* scipy
* scikit-learn
* tqdm

## (Optional) Virtual environment

May use the virtual environment for easy loading: Enable the virtual environement

        source .venv/bin/activate

# How to run

        python pyqt_launcher.py

# Details 

## Loading Data

The program will immediately alert you to load data. The data is expected to be
in CSV format, with each row representing a single edge/connection between two
voxels in the following format:

$$<i_1>, <j_1>, <k_1>, <i_2>, <j_2>, <k_2>, <value>$$

Where $i_1, j_1, k_1$ and $i_2, j_2, k_2$ are the coordinates of the endpoints
of the connection in $ijk$ coordinate system (from *AFNI*'s *3dmaskdump*), and
$value$ is the weight of the connection. Currently, the framework only support
*thresholded* connections, hence $value$ is not used and assumed to take the
value of $1$.

## FCN and Bundles filtering

The FCN:s and bundles are organized in a tree system. In order to get a plot you have to
first mark the bundles you want to plot (or the button "All" for whole FCNs) and
then you press "Plot Selection" in order to see your selections plotted. The
button "Show All" will immediately plot all FCNs. You can also see your selected
bundles for each network in the right column "Selected Bundles". "Clear All"
will clear the whole plot at any time.

## Centroids

If you go to any FCN you will see that for each bundle there is a button option
titled "Centroids", if you click this button a toggle will appear and the
centroid of the bundle will appear if you select the bundle. The centroid of the
bundle represents the path between the average coordinate of the endpoints of
each side of the bundle, which is a function meant to simplify the plot. For
each FCN there is also a button option titled "Toggle All Centroids", which
toggles all "Centroid"-buttons for the FCN, which will lead to only centroids
appearing for the FCN after you press "Plot selection". The button in the main
frame of the GUI titled "Toggle All Centroids" will toggle the
"Centroid"-buttons for all bundles in all FCNS, so whatever bundle or FCN you
select, you will only get the centroids of the bundles.

## Coloring

Each FCN has a default color so all its bundles are colored the same. You change
the color for a bundle by selecting an FCN, then you will see each bundle has a
button titled "Color". This button will be marked with your selected color and
if it is color blanc it means that the bundle has the FCN's default color.

## Fine Tuning

For each bundle there is a button titled "Fine Tune", which is a feature
acivating the graphing parameter buttons "Curve", "Thickness". These buttons
adjust the parameters they are titled after for each bundle and you can see
their settings  when you press "Fine Tune". You can Fine Tune as well for whole
FCNs and as well for all FCNS at once with the buttons "Fine Tune FCN" and "Fine
Tune All FCNs". The endpoint size depends on the voxel-size in your dataset, 
therefore it is best to not adjust the endpoint size in the GUI.

## Export to GIFs

In order to export the given plot, make sure to have pressed plot selection
before. You can adjust the elevation angle of the "camera" pointed at the plot
with the slider labeled as "Elevation", it ranges between -60 and 60 degrees.
There is some slight vertical oscilation (5 degrees amplitude) in the plot in
order to hinder distortion and lagging. If you want to see the animation of the
GIF before exporting it, press "Live preview". The mouse does not contribute to
the angle adjustments of the plot in the GIF-animation, adjustments are made by
the slider. "Export All GIFs" will export one GIF of each FCN all at once.

## Dendrogram

"Show Dendrogram" will show a dendrogram of the hierarchical clusterings of the bundles. The titles of these are colored after their FCN-colors, as marked in the upper corner.