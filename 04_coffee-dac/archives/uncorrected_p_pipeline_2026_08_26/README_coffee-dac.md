# COFFEE-DAC 

COnnectivity-based Functional network Formation, Extraction, and Exploration -
using Dual Agglomerative Clustering


# Introduction

Network-based analysis methods for functional connectivity on functional MRI
data typically operates on nodes rather than edges ([CITE]). Also,
dimensionality reduction through stratifying the data into functional regions,
or ROIs, according to functional atlases is done prior to network construction
to ease computation load ([CITE]).

There are two main issues with this type of approach: Firstly, since there are
currently no atlas that is perfect for every occassion. The entire analysis
pipeline will be biased towards the bias of the atlas chosen.  Furthermore,
implicitly stratifying data into a chosen functional atlas weakens the
data-driven aspect of the analysis.

Second, the definition of functional connectivity is most natural when defined
between nodes, whether they represent voxels, ROIs, or brain regions.
Allocating functional connectivity values to single nodes is not entirely
intuitive, and enforcing this invariably leads to diminished intuitive meaning,
and explicit information is lost.

Typically, functional connectivity measures and their values are associated to
a spatial location, either assigned to individual voxels or ROIs (groups of
voxels). This is done to facilitate interpretability of visualizations of
analysis results since connectivity matrices and none spatially specific graph
visualization methods such as node-ring may be difficult to interpret
intuitively as they lack spatial coordination. Graph visualization methods such
as node-links, which includes inferration of spatial coodinates are easily
overwhelmed visually by execessive number of nodes are connections.


## Aim/Purpose

We aim to develop a analysis pipeline that remedies aforementioned aspects. The
proposed pipeline will be completely data-driven without the need to stratify
data prior into ROIs using functional atlases, instead opting for voxel-based
analysis, and will use functional connectivity measures solely as defined as
values between pairs of voxels.

Here we present a higher-level analysis pipeline for organizing and visualizing
voxel-based functional connectivity, defined between pairs of voxels, results
from second-level group statistics. 

# Materials and Methods

Our pipeline consists primary of 2 parts: 1) Stratify the surviving edges into
"connectivity bundles" to facilitate visualization. 2) Visualize connectivity
between nodes in a intuitive manner.

As a proof-of-concept, a previously acquired dataset was used to generate a
preliminary voxel-wise functional connectivity matrix group statistics.

# What is COFFEE-DAC

COFFEE-DAC is an analysis pipeline for (resting-state) functional MRI data,
consisting of a collection of python routines in order extract resting-state
functional connectivity networks (FCN) from functional MRI data.

COFEE-DAC operates essentially on functional connectivity information (i.e.
relationship between two brain locations), rather than attempting to convert
this information into metrics which can be assigned to a single brain location.
This makes COFEE-DAC unique as it performs clustering on relationships between
data-points rather than the traditional approach of clustering data points
based on their relationships.

Given a set of functional connectivity/edge in the form of the coordinates of
their end-points, COFFEE-DAC will be able to:

1. Perform edge clustering/bundling in order to obtain cluster bundles which
   are tightly associated with one another. 

2. Construct FCNs (and their sub-networks), through merging edge bundles
   together such that all parts of a FCN is connected with one another (within
reason).

# COFEE-DAC algorithm

Agglomerative (hierarchical) clustering (I).

* Complete-linkage. 
* Distance (between edges): Maximum distance between any end-point of one edge
  to any end-point of the other.

The minimum cut-level producing an maximum max-min metric falling below a
certain threshold is determined to be the recommended number of resulting edge
bundles. Threshold is typically relaxed to 2-3 voxels worth of max-min distance.

Agglomerative (hierarchical) clustering (II).

* Single-linkage
* Distance (between bundles): Minimum distance between any edge in a bundle
  with any edge in another bundle. 

"Optimal" cut levels are determined either using resulting clusters' edge count
(or edge bundle count). Alternatively one may simply visualize the entire
dendrogram as the number leaves (edge bundles) should be sufficiently small for
clear overview of the data at this stage. Two main options are available at
this stage:

1. Single-linkage when isolating main networks(s). Cut-level is taken as the
   cut right before edge count of the largest cluster drops drastically.

2. Average-linkage to divide network(s) into sub-networks. Depending on number
   of sub-divisions of your network (N), set cut-level one after the Nth
significant drop in maximum cluster edge count.

# Technical details

## Hierarchical clustering

Hierarchical clustering has the following advantages against other clustering
algorithms such as DBSCAN.

1. Forces all edges to belong to a bundle/FCN. Assuming statistical corrections
   and appropriate noise filtering is performed prior, avoiding discarding
connections is desirable in this application.

2. May apply non-euclidean distance metrics as basis of clustering.

3. Linkage can be tweaked to adjust clustering results further.

The downside is that clustering results will depend heavily on the selection of
number of output clusters. This is not desirable as we do not know exactly the
number of edge bundles exist in the dataset prior. However, this issues might
not be detrimental so long all edges are preserved in the first clustering
stage to perform FCN formation upon.

## Edge bundles

COFFEE-DAC uses the following criterion to bundle edges together:

* Two edges are closedly associated when they have similar end-points. The
  edges may even share a single end-point.

* Even if one of their endpoints are the same, two edges do not belong to the
  same edge bundle if their other endpoint is (sufficiently) far away from one
another. 

In this step, the algorithm is weighted more towards avoiding bundling
unrelated edge rather than forming FCNs, so even if the latter point is
actually a desired trait of FCNs, it is avoided in this step 

For distance metric computations, maximum end-point distance is theoretically
the natural option for distance. Together with complete-linkage hierarchical
clustering (default), is the most appropriate in accordance to the
aforementioned criterion. 

### Note on average end-point distance and linkage

Average end-point distance is also practically
valid option. Average end-point distance relax edge bundle criterion slightly
in order to perform more aggressive dimensionaltiy reduction at the risk of
bundling different edge bundles together. Since the end purpose is to form
FCNs, and the risk of said edge bundles actually belonging to different
FCNs is minimal.

Similar argument is valid for the hierarchical clustering linkage scheme, with
the theoretical default being "complete". However in practice using
average-linkage produces similar results to give well separated edge bundles,
and whenever it doesn not, the resulting "merged" edge bundles are so closely
associated with one another that they seemingly cannot belong to different FCNs
upon visual inspection.

## Max-min metric

We can define a set of criteria in order to determine the optimal cut-level for
edge bundling.

To minimize the number of output edge bundles, while retaining clear separation
between unrelated edges, we define the following metric to determine whether a
set of edges are closely bundled:

The minimum distance between all edges with all other edges in the bundle is
calculated for all pairs of edges in a bundle. The maximum of the minimum
distances is then taken as this provides information on the maximum spatial
deviation between closest point of contant between edge connections (max-min
metric). 

To determine a suitable cut-level for number of edge bundles in the first
clustering stage, the cut level is systematically increased. At least cut
level, the max-min metric for all produced edge bundles. The minimum cut-level
producing an maximum max-min metric falling below a certain threshold is a good
choice for number of resulting edge bundles. Different thresholds can be set,
with a distance of sqrt(3) being neighbours (at least corners touching),
sqrt(12) allows for one voxel's space worth of separation, and sqrt(37) allows
for 2 voxels worth of distance, and etc. This threshold have the advantage of
of being easy to visualize and intuitive. Typically the threshold used is
relaxed to 2-3 voxels worth of max-min separation.

## Construct FCNs from edge bundles.

The following criteron for a FCN is used in their formation:

* A FCN consists may be spatially distributed, as long as there is a path of
  edges (directly or indirectly) between all brain regions in the network.

* Two fiber bundles are belong in the same fiber bundle as long as the closest
  point between two fiber bundle (or fiber bundle clusters) are close to each
other, corresponding to single linkage.

Hierarchical clustering is also opted here in order to agglomeratively
construct networks from edge bundles. However, due to the different purposes of
clustering, this second application of the algorithm uses single-linkage with
minimum distances between all edges in each pair of edge bundles as distance
metric.

# Discussion

There are issues with dimensionality with an edge-based analysis method as
opposed to a node based approach, since the potential number of edges is
squared that of number of nodes. 

# Notes on test dataset

## Haldol dataset

40 healthy control subjects, which each subject scanned twice, once with
Haloperidol (2mg) and once with placebo injested immediately prior to scanning. 

## data acquisition

Resting-state fMRI data acquired using Siemens Tim Trio (Erlangen, Germany)
using  single-shot 2D gradient-recalled echo echo-planar imaging pulse sequence
with the following acquisition parameters: 32 transverse slices, slice
thickness = 2.6 mm, TR/TE = 2000/35 ms, FOV = 220 mm, matrix size = 64 x 64, FA
= 90, 300 time-frames, IPAT = 2. A 32-channel phased-array head coil was used
for the signal reception.

## data preprocessing

The resting-state fMRI datasets were preprocessed using AFNI
(http://afni.nimh.nih.gov/afni) and FSL (http://www.fmrib.ox.ac.uk/fsl)
programs, with a bash wrapper shell. The pipeline includes:

* removal of the first 10 timeframes

* temporal de-spiking

* removing baseline trends up to the third-order polynomial, effective
  band-pass filtering was performed with a low-pass filter at 0.08 Hz.

* motion correction using a six-parameter rigid body image registration

* average volume brain masking

* spatial normalization to the MNI template using a 12-parameter affine
  transformation and mutual-information cost function

* spatial smoothing/resampling to isotropic resolution using a 4 mm FWHM
  Gaussian kernel

* Nuisance signal removal involved voxel-wise regression with 16 regressors
  based on motion correction parameters, average ventricle and core white
  matter signals, and their first-order derivatives.

## preliminary data analysis

The Pearson's cross-correlation coefficient were calculated for all voxel size
the brain-mask against all other voxels, creating a correlation matrix for each
subject using the AFNI program 3dAutoTcorrelate, which stores the correlation
values in a 3D+t nifti dataset, where each "time-point" corresponds to a single
voxels correlation against all other voxels, with a total of N "time-points"
where N is the number of voxels inside the brain mask.

For each element in the correlation matrix t-tests were made between placebo
and drug groups and the 1% highest t-scores were kept as significant
differences between the groups and binarized to 1.

