"""
Plot segmentations from the fluorescence extractor pipeline for debugging.

Uses the EP (nucleus + cell body) segmentation. Only the first 3 timepoints
are visualised.

Run from Repressilator_tests directory:
    python -m tests.plot_segmentations

Saves plots to tests/segmentation_plots/ (one per timepoint).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
import repressilator_analysis as ra
from repressilator_analysis import fluorescence_extraction_EP


def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repressilator_root = os.path.dirname(test_dir)
    intensity_dir = os.path.join(repressilator_root, "images", "intensity")
    phase_dir = os.path.join(repressilator_root, "images", "phase")

    if not os.path.isdir(phase_dir):
        print(f"Phase image dir not found: {phase_dir}")
        print("Run from Repressilator_tests or ensure images/phase exists.")
        return

    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )

    # EP segmentation: nucleus (darkest) + cell body (lighter), min_cell_area=5
    tracks, labeled_images = fluorescence_extraction_EP.track_cells_across_time(
        phase_images, min_cell_area=5
    )

    out_dir = os.path.join(test_dir, "segmentation_plots")
    os.makedirs(out_dir, exist_ok=True)

    # Only visualise first 3 timepoints
    n_frames = 1
    expected_cells = 80

    for i in range(n_frames):
        phase = phase_images[i]
        labeled = labeled_images[i]
        three_class = fluorescence_extraction_EP.multiotsu_three_classes(phase)
        n_cells = len(tracks[i])

        # Clumps = connected components of cell mask before watershed (touching cells as one)
        clumps = fluorescence_extraction_EP.get_cell_clumps(phase, min_cell_area=5)

        # Nuclei only (same pipeline as segment_cells) to check if we get ~80
        nuclei_labeled = fluorescence_extraction_EP.get_nuclei_labeled(phase, min_cell_area=5)
        n_nuclei = nuclei_labeled.max()

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        # 1) Phase image only
        ax = axes[0]
        ax.imshow(phase, cmap="gray")
        ax.set_title(f"Frame {i} (t={timepoints[i]:.0f}m)\nPhase")
        ax.axis("off")

        # 2) Clumps: cell material as connected components (before splitting)
        ax = axes[1]
        n_clumps = clumps.max()
        clump_cmap = plt.colormaps["nipy_spectral"].resampled(max(n_clumps + 1, 2))
        clump_overlay = clump_cmap(clumps / max(n_clumps, 1))
        clump_overlay[clumps == 0, -1] = 0
        ax.imshow(phase, cmap="gray", alpha=0.5)
        ax.imshow(clump_overlay)
        ax.set_title("Clumps (before split)\nconnected cell regions")
        ax.axis("off")

        # 3) Multi-Otsu 3 classes: two intensities (background, cell body, nucleus)
        ax = axes[2]
        class_cmap = ListedColormap(["#2d2d2d", "#7eb87e", "#e74c3c"])  # grey, green, red
        ax.imshow(three_class, cmap=class_cmap, vmin=0, vmax=2)
        ax.set_title("Multi-Otsu (3 classes)\n0=bg, 1=cell body, 2=nucleus")
        ax.axis("off")

        # 4) Nuclei only: largest intensities (class 2) as labeled components
        ax = axes[3]
        ax.imshow(phase, cmap="gray", alpha=0.6)
        nuc_cmap = plt.colormaps["nipy_spectral"].resampled(max(n_nuclei + 1, 2))
        nuc_overlay = nuc_cmap(nuclei_labeled / max(n_nuclei, 1))
        nuc_overlay[nuclei_labeled == 0, -1] = 0
        ax.imshow(nuc_overlay)
        ax.set_title(f"Nuclei only (darkest intensity)\n{n_nuclei} nuclei (expected {expected_cells})")
        ax.axis("off")

        # 5) Watershed splits: one cell per nucleus (same nuclei as panel 4)
        ax = axes[4]
        ax.imshow(phase, cmap="gray")
        labeled = fluorescence_extraction_EP.segment_cells(phase, min_cell_area=5)
        n_cells_labeled = labeled.max()
        # Normalize so each label gets a distinct color (use N+1 so max label is < 1)
        n_colors = max(n_cells_labeled + 1, 2)
        label_cmap = plt.colormaps["nipy_spectral"].resampled(n_colors)
        label_overlay = label_cmap(labeled.astype(float) / n_colors)
        label_overlay[labeled == 0, -1] = 0
        ax.imshow(label_overlay)
        ax.set_title(f"Watershed splits (nucleus-based)\n{n_cells_labeled} cells (expected {expected_cells})")
        ax.axis("off")

        plt.suptitle(f"Frame {i}: {n_cells} cells detected (expected {expected_cells})", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}  (nuclei: {n_nuclei}, cells: {n_cells_labeled}, expected: {expected_cells})")

    print(f"\nDone. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
