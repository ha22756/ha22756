"""
Run ADT segmentation on all 8 nodules in roi_exam.mat and display/save results.

Requirements:
    pip install numpy scipy matplotlib scikit-image

This script produces:
  - 8 segmentation figures
  - 8 ARG-vs-T0 plots
  - 1 threshold surface image
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adt_segmentation import adt_segmentation, create_distance_array
from display_masks import display_masks


def load_roi_exam(mat_path: str | Path):
    data = sio.loadmat(mat_path, simplify_cells=True)
    return data["roi_exam"]


def main():
    mat_path = Path("roi_exam.mat")
    output_dir = Path("adt_python_outputs")
    output_dir.mkdir(exist_ok=True)

    roi_exam = load_roi_exam(mat_path)

    # In the original MATLAB project the cue is approximately (44,44) in 1-based indexing.
    # In Python that becomes (43,43) in 0-based indexing.
    cue = (43, 43)
    rm = 25
    TD = 1.7
    TO_values = np.arange(-2.0, 2.0001, 0.01)

    d = create_distance_array(shape=(87, 87), cue=cue)

    # Save one threshold example at T0 = 0
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    rows, cols = np.indices((87, 87))
    T0 = 0.0
    T_example = T0 + TD * ((1 - np.exp(-d / (rm**2))) / (1 - np.exp(-1)))
    T_example[d >= rm**2] = np.inf
    finite_T = np.where(np.isfinite(T_example), T_example, np.nan)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(cols, rows, finite_T, linewidth=0, antialiased=True)
    ax.set_title("Distance-based threshold function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T(x,y)")
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_surface.png", dpi=200)
    plt.close(fig)

    for i, item in enumerate(roi_exam, start=1):
        x = np.asarray(item["cxr_contrast"], dtype=float)
        lung_mask = np.asarray(item["lung_mask"]).astype(bool)

        mask, avg_scores, TO_best, thresh = adt_segmentation(
            x=x,
            lung_mask=lung_mask,
            cue=cue,
            rm=rm,
            TD=TD,
            TO_values=TO_values,
            d=d,
        )

        # Segmentation result
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(x, cmap="gray")
        display_masks(mask, colors=['g'], line_widths=[2.0], ax=ax)
        ax.plot(cue[0], cue[1], "b+", markersize=10, markeredgewidth=2)
        ax.contour(thresh, levels=10, colors='r', linewidths=0.7, alpha=0.6)
        ax.set_title(f"Nodule {i} - ADT segmentation")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / f"nodule_{i:02d}_segmentation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ARG plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(TO_values, avg_scores)
        ax.set_xlabel("T0")
        ax.set_ylabel("ARG(T0)")
        ax.set_title(f"Average radial gradient vs T0 - nodule {i}")
        best_idx = int(np.argmax(avg_scores))
        ax.plot(TO_values[best_idx], avg_scores[best_idx], "ro")
        ax.annotate("max ARG", (TO_values[best_idx], avg_scores[best_idx]))
        fig.tight_layout()
        fig.savefig(output_dir / f"nodule_{i:02d}_arg_curve.png", dpi=200)
        plt.close(fig)

        print(f"Processed nodule {i}: TO_best={TO_best:.3f}")

    print(f"Done. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
