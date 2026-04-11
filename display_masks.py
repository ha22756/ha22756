"""
Display one or more binary mask contours on the current Matplotlib axes.
Converted from display_masks.m.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def display_masks(mask, colors=None, line_widths=None, ax=None):
    if ax is None:
        ax = plt.gca()

    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[:, :, None]

    num_masks = mask.shape[2]

    if colors is None:
        base_colors = ['r', 'g', 'b', 'm', 'c', 'y']
        colors = [base_colors[i % len(base_colors)] for i in range(num_masks)]

    if line_widths is None:
        line_widths = [1.0] * num_masks

    for i in range(num_masks):
        ax.contour(mask[:, :, i].astype(float), levels=[0.5], colors=[colors[i]], linewidths=[line_widths[i]])
