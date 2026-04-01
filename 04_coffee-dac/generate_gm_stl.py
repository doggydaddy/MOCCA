#!/usr/bin/env python3
"""
generate_gm_stl.py
==================
Generate brain3mm_gm.stl – a detailed GM/CSF surface with gyri and sulci –
from the high-resolution (1 mm isotropic) brain.nii.gz template.

Strategy
--------
1. Load brain.nii.gz  (256³, 1 mm iso, MNI152-style T1).
2. Lightly smooth (σ=1 voxel) to reduce marching-cubes staircase artefacts
   while preserving gyral folds.
3. Run marching cubes at a threshold that sits at the GM/CSF boundary
   (~30 intensity units on this 0-119 scale) to capture the full outer
   cortical surface including all sulci.
4. Transform the resulting vertex coordinates from 1mm-voxel index space
   → world (MNI) space via the 1mm affine, then
   → 3mm-voxel index space via the inverse of the 3mm affine.
   This puts the mesh in the same coordinate frame as the edge data loaded
   by the MOCCA GUI (which indexes into brain3mm.nii voxels).
5. Decimate and smooth the mesh for a clean, still-detailed surface.
6. Save as brain3mm_gm.stl next to the other STL files.

Usage
-----
    cd /mnt/islay/MOCCA/04_coffee-dac
    .venv/bin/python generate_gm_stl.py
"""

import os
import sys
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, '..', 'templates')

SRC_1MM  = os.path.join(TEMPLATE_DIR, 'brain.nii.gz')
REF_3MM  = os.path.join(TEMPLATE_DIR, 'brain3mm.nii')
OUT_STL  = os.path.join(SCRIPT_DIR,   'brain3mm_gm.stl')

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
# Gaussian pre-smooth σ (in 1mm voxels). 0.5 tames staircase noise while
# preserving gyral folds better than the previous σ=1.0.
SMOOTH_SIGMA = 0.5

# Marching-cubes isovalue at the GM/CSF boundary.
# The 1mm template runs 0-119; WM peak ~108-114, GM ~60-105, CSF/BG ≤30.
# A level of ~30 captures the full outer cortical surface.
MC_LEVEL = 30.0

# Post-MC surface operations (applied in PyVista)
# Decimation target – keep this fraction of triangles (0-1).
# 0.40 retains ~40% of raw MC triangles, preserving gyri/sulci detail.
# (Previous value was 0.08 which destroyed most surface detail.)
DECIMATE_TARGET = 0.40

# Laplacian smoothing passes on the decimated mesh.
# 15 passes is enough to remove MC staircase without blurring gyral folds.
SMOOTH_PASSES = 15

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading templates...")
img_1mm = nib.load(SRC_1MM)
img_3mm = nib.load(REF_3MM)

data_1mm   = img_1mm.get_fdata(dtype=np.float32)
affine_1mm = img_1mm.affine.astype(np.float64)
affine_3mm = img_3mm.affine.astype(np.float64)

print(f"  1mm volume : {data_1mm.shape},  affine diagonal: "
      f"{np.diag(affine_1mm[:3,:3]).round(2)}")
print(f"  3mm volume : {img_3mm.shape},  affine diagonal: "
      f"{np.diag(affine_3mm[:3,:3]).round(2)}")

# ---------------------------------------------------------------------------
# Smooth then marching cubes
# ---------------------------------------------------------------------------
print(f"Gaussian pre-smoothing (σ={SMOOTH_SIGMA})...")
data_smooth = gaussian_filter(data_1mm, sigma=SMOOTH_SIGMA)

print(f"Marching cubes at level={MC_LEVEL}...")
verts_ijk1, faces, normals, _ = marching_cubes(
    data_smooth,
    level=MC_LEVEL,
    spacing=(1.0, 1.0, 1.0),   # 1mm isotropic; we'll apply the real affine below
    allow_degenerate=False,
    step_size=1,
)
print(f"  Raw mesh: {len(verts_ijk1):,} vertices, {len(faces):,} triangles")

# ---------------------------------------------------------------------------
# Coordinate transform:  1mm voxel-index  →  world (MNI)  →  3mm voxel-index
# ---------------------------------------------------------------------------
print("Transforming vertices to 3mm voxel-index space...")

# Homogeneous 1mm voxel → world
ones     = np.ones((len(verts_ijk1), 1), dtype=np.float64)
verts_h  = np.hstack([verts_ijk1.astype(np.float64), ones])   # (N, 4)
verts_world = (affine_1mm @ verts_h.T).T[:, :3]               # (N, 3)

# World → 3mm voxel-index
inv_aff3 = np.linalg.inv(affine_3mm)
verts_h2 = np.hstack([verts_world, ones])
verts_ijk3 = (inv_aff3 @ verts_h2.T).T[:, :3]                 # (N, 3)

# ---------------------------------------------------------------------------
# Build PyVista mesh
# ---------------------------------------------------------------------------
# pyvista faces format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
n_faces   = len(faces)
pv_faces  = np.hstack([
    np.full((n_faces, 1), 3, dtype=np.int64),
    faces.astype(np.int64)
]).ravel()

mesh = pv.PolyData(verts_ijk3.astype(np.float32), pv_faces)

# ---------------------------------------------------------------------------
# Decimate  →  smooth
# ---------------------------------------------------------------------------
print(f"Decimating to {DECIMATE_TARGET*100:.0f}% of triangles...")
mesh = mesh.decimate(1.0 - DECIMATE_TARGET)
print(f"  After decimation: {mesh.n_points:,} vertices, {mesh.n_cells:,} triangles")

print(f"Laplacian smoothing ({SMOOTH_PASSES} passes)...")
mesh = mesh.smooth(n_iter=SMOOTH_PASSES, relaxation_factor=0.05)

# Recompute clean normals
mesh = mesh.compute_normals(cell_normals=False, point_normals=True, consistent_normals=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"Saving  →  {OUT_STL}")
mesh.save(OUT_STL)
print("Done.")

# Quick sanity check: print bbox in 3mm voxel space
pts = np.array(mesh.points)
print(f"\nVertex bbox (3mm voxel-index space):")
print(f"  X: {pts[:,0].min():.1f} .. {pts[:,0].max():.1f}")
print(f"  Y: {pts[:,1].min():.1f} .. {pts[:,1].max():.1f}")
print(f"  Z: {pts[:,2].min():.1f} .. {pts[:,2].max():.1f}")
print(f"\nFor reference, brain3mm.nii grid size: {img_3mm.shape}")
