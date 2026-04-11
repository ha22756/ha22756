"""
Average radial gradient over an entire mask.
Converted from radial_gradient_all.m.
"""
from __future__ import annotations

import numpy as np


def radial_gradient_all(gradx: np.ndarray, grady: np.ndarray, mask: np.ndarray, cue=(43, 43)):
    gradx = np.asarray(gradx, dtype=float)
    grady = np.asarray(grady, dtype=float)
    mask = np.asarray(mask).astype(bool)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return -np.inf, np.nan

    grad_vectors = np.column_stack((-gradx[ys, xs], -grady[ys, xs]))

    cx, cy = cue
    radial_vectors = np.column_stack((xs - cx, ys - cy))
    mags = np.sqrt(np.sum(radial_vectors**2, axis=1))
    mags[mags == 0] = 1.0
    radial_unit_vectors = radial_vectors / mags[:, None]

    inner_products = np.sum(grad_vectors * radial_unit_vectors, axis=1)
    return float(np.mean(inner_products)), float(np.std(inner_products))
