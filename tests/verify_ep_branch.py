"""
Verify you're on the EP branch and that the EP segmentation code is being used.

Run from Repressilator_tests:
    python -m tests.verify_ep_branch

Prints: branch, which segment_cells is used by EP vs original, and cell count
from one frame using EP.
"""

import os
import sys
import subprocess
import numpy as np

# 1) Git branch
try:
    r = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
    )
    branch = (r.stdout or "").strip() or "(unknown)"
except Exception as e:
    branch = f"(git error: {e})"
print("Git branch:", branch)
print()

# 2) Which module provides segment_cells
import repressilator_analysis as ra
from repressilator_analysis import fluorescence_extraction_EP

orig_segment = ra.fluorescence_extraction.segment_cells
ep_segment = fluorescence_extraction_EP.segment_cells

print("Original fluorescence_extraction.segment_cells:")
print("  ", orig_segment.__module__, "->", getattr(orig_segment, "__doc__", "")[:60].strip() or "(no doc)")
print("  file:", getattr(orig_segment, "__code__", None) and orig_segment.__code__.co_filename)
print()
print("EP fluorescence_extraction_EP.segment_cells:")
print("  ", ep_segment.__module__, "->", getattr(ep_segment, "__doc__", "")[:60].strip() or "(no doc)")
print("  file:", getattr(ep_segment, "__code__", None) and ep_segment.__code__.co_filename)
print()

# 3) What the plot script and EP test use
print("Plot script and EP test use: fluorescence_extraction_EP (segment_cells from EP file above)")
print()

# 4) One-frame EP segmentation and cell count
test_dir = os.path.dirname(os.path.abspath(__file__))
repressilator_root = os.path.dirname(test_dir)
phase_dir = os.path.join(repressilator_root, "images", "phase")
if os.path.isdir(phase_dir):
    timepoints, _, phase_images = ra.image_loader.load_timeseries(
        os.path.join(repressilator_root, "images", "intensity"),
        phase_dir,
    )
    labeled = fluorescence_extraction_EP.segment_cells(phase_images[0], min_cell_area=5)
    n_cells = len(np.unique(labeled[labeled > 0]))
    print("EP segment_cells on first frame:")
    print("  number of cells (labels):", n_cells)
    print("  max label:", labeled.max())
else:
    print("images/phase not found; skipping one-frame EP run")
print()
print("If branch is ep/nucleus-cell-segmentation and EP file path contains fluorescence_extraction_EP, you are using the new code.")
