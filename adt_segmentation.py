"""
Adaptive Distance-Based Threshold (ADT) lung nodule segmentation in Python.

Converted from the original MATLAB implementation and assignment.
The function searches over candidate T0 values, computes the Average
Radial Gradient (ARG) for each candidate mask, and returns the best mask.

Inputs
------
x : ndarray, shape (87, 87)
    Input ROI image (contrast-enhanced chest x-ray patch).
lung_mask : ndarray, shape (87, 87)
    Binary lung mask.
cue : tuple[int, int]
    Cue point in (x, y) image coordinates. For the provided exam data this
    is typically (44, 44) using 1-based MATLAB indexing. In Python we use
    0-based indexing, so the center pixel is usually (43, 43).
rm : float
    Radius parameter for the threshold matrix.
TD : float
    Tuning parameter for the threshold function.
TO_values : ndarray
    Array of candidate T0 offset values to search.
d : ndarray, optional
    Precomputed squared-distance array. If None, it is computed from cue.

Returns
-------
mask : ndarray
    Final binary mask for the best T0.
avg_scores : ndarray
    Average radial gradient values for all tested T0 values.
TO_best : float
    Best threshold offset.
T_best : ndarray
    Threshold matrix corresponding to TO_best.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, opening
from radial_gradient_all import radial_gradient_all


def create_distance_array(shape=(87, 87), cue=(43, 43)):
    rows, cols = shape
    yy, xx = np.indices((rows, cols))
    cx, cy = cue  # cue is (x, y)
    d = (xx - cx) ** 2 + (yy - cy) ** 2
    return d


def threshold_matrix(d: np.ndarray, rm: float, TD: float, T0: float) -> np.ndarray:
    T = T0 + TD * ((1 - np.exp(-d / (rm**2))) / (1 - np.exp(-1)))
    T = T.astype(float)
    T[d >= rm**2] = np.inf
    return T


def extract_component_connected_to_cue(mask: np.ndarray, cue=(43, 43)) -> np.ndarray:
    labeled, _ = ndi.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int))
    cx, cy = cue
    cue_label = labeled[cy, cx]
    if cue_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labeled == cue_label


def adt_segmentation(
    x: np.ndarray,
    lung_mask: np.ndarray,
    cue=(43, 43),
    rm: float = 25.0,
    TD: float = 1.7,
    TO_values: np.ndarray | None = None,
    d: np.ndarray | None = None,
):
    if TO_values is None:
        TO_values = np.arange(-2.0, 2.0001, 0.01)

    x = np.asarray(x, dtype=float)
    lung_mask = np.asarray(lung_mask).astype(bool)

    if d is None:
        d = create_distance_array(x.shape, cue=cue)

    # Python gradients: gy is along rows (y), gx is along cols (x)
    grady, gradx = np.gradient(x)

    avg_scores = np.zeros(len(TO_values), dtype=float)

    for i, t0 in enumerate(TO_values):
        T = threshold_matrix(d, rm, TD, t0)
        Y = x > T
        mask1 = Y & lung_mask
        nodule = ndi.binary_fill_holes(mask1)
        nodule_open = opening(nodule, footprint=disk(1))
        candidate = extract_component_connected_to_cue(nodule_open, cue=cue)
        avg_scores[i], _ = radial_gradient_all(gradx, grady, candidate, cue)

    best_idx = int(np.argmax(avg_scores))
    TO_best = float(TO_values[best_idx])
    T_best = threshold_matrix(d, rm, TD, TO_best)

    Y = x > T_best
    mask1 = Y & lung_mask
    nodule = ndi.binary_fill_holes(mask1)
    nodule_open = opening(nodule, footprint=disk(1))
    mask = extract_component_connected_to_cue(nodule_open, cue=cue)

    return mask.astype(np.uint8), avg_scores, TO_best, T_best
