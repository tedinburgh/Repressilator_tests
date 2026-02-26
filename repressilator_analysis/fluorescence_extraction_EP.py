"""
Fluorescence extraction (EP): segment cells using two intensity levels.

Phase contrast: darker grey = nucleus, lighter grey = cell body.
Nuclei are non-overlapping; we use them as seeds and watershed into the
cell-body region to separate touching/overlapping cells.
"""

import numpy as np
from typing import List, Dict, Tuple
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage


def multiotsu_three_classes(phase_image: np.ndarray) -> np.ndarray:
    """
    Return the 3-class Multi-Otsu segmentation for display.

    Pixel values: 0 = background (lightest), 1 = cell body (middle),
    2 = nucleus (darkest). No morphology or watershed applied.

    Args:
        phase_image: Phase contrast image (grayscale or RGB)

    Returns:
        Labeled image with values in {0, 1, 2} (same shape as input).
    """
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2).astype(np.float64)
    else:
        gray = np.asarray(phase_image, dtype=np.float64)

    try:
        thresholds = filters.threshold_multiotsu(gray, classes=3)
        t0, t1 = float(thresholds[0]), float(thresholds[1])
        # 0 = background, 1 = cell body, 2 = nucleus
        out = np.zeros_like(gray, dtype=np.uint8)
        out[gray >= t1] = 0
        out[(gray >= t0) & (gray < t1)] = 1
        out[gray < t0] = 2
        return out
    except (ValueError, IndexError):
        th = filters.threshold_otsu(gray)
        out = np.zeros_like(gray, dtype=np.uint8)
        out[gray < th] = 2  # treat as nucleus class for display
        out[gray >= th] = 0
        return out


def get_cell_clumps(phase_image: np.ndarray, min_cell_area: int = 50) -> np.ndarray:
    """
    Return labeled connected components of the cell mask (nucleus + cell body)
    before watershed splitting. For visualisation: "clumps" = touching cells as one.

    Pixel values: 0 = background, 1..N = clump label (each connected component).
    """
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2).astype(np.float64)
    else:
        gray = np.asarray(phase_image, dtype=np.float64)
    try:
        thresholds = filters.threshold_multiotsu(gray, classes=3)
        t1 = float(thresholds[1])
        cell_mask = gray < t1
    except (ValueError, IndexError):
        th = filters.threshold_otsu(gray)
        cell_mask = gray < th
    cell_mask = morphology.remove_small_objects(cell_mask, max_size=min_cell_area)
    cell_mask = morphology.remove_small_holes(cell_mask, max_size=min_cell_area)
    clumps = measure.label(cell_mask)
    clumps = segmentation.clear_border(clumps)
    return clumps


def get_nuclei_labeled(phase_image: np.ndarray, min_cell_area: int = 50) -> np.ndarray:
    """
    Return labeled connected components of the nucleus mask (darkest intensity class).

    Uses the same Multi-Otsu and cleaning as segment_cells, so the count matches
    what the watershed pipeline uses. For debugging: plot this to see how many
    nuclei are detected (expected one per cell, e.g. 80).

    Returns:
        Labeled image: 0 = background, 1..N = nucleus label (same shape as input).
    """
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2).astype(np.float64)
    else:
        gray = np.asarray(phase_image, dtype=np.float64)
    try:
        thresholds = filters.threshold_multiotsu(gray, classes=3)
        t0 = float(thresholds[0])
        nucleus_mask = gray < t0
    except (ValueError, IndexError):
        th = filters.threshold_otsu(gray)
        nucleus_mask = gray < th
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, max_size=max(min_cell_area // 5, 3))
    nucleus_mask = morphology.remove_small_holes(nucleus_mask, max_size=max(min_cell_area // 5, 3))
    nuclei_labeled = measure.label(nucleus_mask)
    nuclei_labeled = segmentation.clear_border(nuclei_labeled)
    nuclei_labeled = morphology.remove_small_objects(nuclei_labeled, max_size=max(min_cell_area // 10, 2))
    return nuclei_labeled


def segment_cells(phase_image: np.ndarray, min_cell_area: int = 50) -> np.ndarray:
    """
    Segment individual cells using nucleus (darkest) and cell body (lighter) intensities.

    Uses exactly the same nuclei as the 3-class plot (class 2 = nucleus), from
    get_nuclei_labeled(), so one cell per detected nucleus (e.g. 79 cells from 79 nuclei).
    One watershed over the full cell mask with all nuclei as seeds assigns every
    cell pixel to the nearest nucleus; boundaries run along the equidistant lines.

    Args:
        phase_image: Phase contrast image (grayscale or RGB)
        min_cell_area: Minimum cell area in pixels

    Returns:
        Labeled image where each cell has a unique integer label
    """
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2).astype(np.float64)
    else:
        gray = np.asarray(phase_image, dtype=np.float64)

    # Nuclei = same as plot (class 2 / get_nuclei_labeled) so we get e.g. 79
    nuclei_labeled = get_nuclei_labeled(phase_image, min_cell_area)

    # Cell mask (nucleus + cell body); do not clear_border so we keep all cell material
    try:
        thresholds = filters.threshold_multiotsu(gray, classes=3)
        t1 = float(thresholds[1])
        cell_mask = gray < t1
    except (ValueError, IndexError):
        th = filters.threshold_otsu(gray)
        cell_mask = gray < th
    cell_mask = morphology.remove_small_objects(cell_mask, max_size=min_cell_area)
    cell_mask = morphology.remove_small_holes(cell_mask, max_size=min_cell_area)

    nucleus_props = list(measure.regionprops(nuclei_labeled))
    if not nucleus_props:
        # No nuclei: fallback to one region
        result = measure.label(cell_mask)
        result = segmentation.clear_border(result)
        return measure.label(result > 0)

    # One marker per nucleus (same labels as nuclei_labeled: 1..N)
    centres_binary = np.zeros_like(gray, dtype=np.uint8)
    markers = np.zeros(gray.shape, dtype=np.intp)
    used_pixels = set()
    for prop in nucleus_props:
        r, c = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        r, c = np.clip(r, 0, gray.shape[0] - 1), np.clip(c, 0, gray.shape[1] - 1)
        if (r, c) in used_pixels:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < gray.shape[0] and 0 <= c2 < gray.shape[1] and (r2, c2) not in used_pixels:
                    r, c = r2, c2
                    break
        used_pixels.add((r, c))
        centres_binary[r, c] = 1
        markers[r, c] = prop.label

    # Relief = negative distance to nearest nucleus centre → split along equidistant lines
    dist_to_nearest_nucleus = ndimage.distance_transform_edt(centres_binary)
    relief = -dist_to_nearest_nucleus.astype(np.float64)

    result = segmentation.watershed(relief, markers, mask=cell_mask)

    # Fill any mask pixels left as 0 by watershed
    unlabeled = (result == 0) & cell_mask
    if np.any(unlabeled):
        while np.any(unlabeled):
            for prop in nucleus_props:
                lid = prop.label
                region = result == lid
                grown = morphology.binary_dilation(region) & unlabeled
                result[grown] = lid
                unlabeled = unlabeled & ~grown

    # Clear border: set border-touching cells to 0 (do not merge the rest)
    result = segmentation.clear_border(result)
    # Compact remaining labels to 1..N (do NOT use measure.label(result > 0) -
    # that would merge all non-zero pixels into one connected component)
    unique_labels = np.unique(result)
    unique_labels = unique_labels[unique_labels > 0]
    new_result = np.zeros_like(result, dtype=np.intp)
    for new_id, old_id in enumerate(unique_labels, start=1):
        new_result[result == old_id] = new_id
    return new_result


def extract_cell_fluorescence(
    intensity_image: np.ndarray,
    labeled_cells: np.ndarray,
    channels: List[str] = ['red', 'green'],
) -> Dict[int, Dict[str, float]]:
    """
    Extract fluorescence values for each cell and channel.

    Args:
        intensity_image: RGB fluorescence image
        labeled_cells: Labeled cell image from segmentation
        channels: List of color channels to extract ('red', 'green', 'blue')

    Returns:
        Dictionary mapping cell_id -> {channel: mean_intensity}
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}

    cell_ids = np.unique(labeled_cells)
    cell_ids = cell_ids[cell_ids > 0]

    results = {}

    for cell_id in cell_ids:
        cell_mask = labeled_cells == cell_id
        cell_data = {}

        for channel in channels:
            if channel.lower() not in channel_map:
                continue

            channel_idx = channel_map[channel.lower()]
            if intensity_image.ndim == 3:
                channel_image = intensity_image[:, :, channel_idx]
            else:
                channel_image = intensity_image

            cell_fluorescence = channel_image[cell_mask]
            cell_data[channel] = float(np.mean(cell_fluorescence))

        results[int(cell_id)] = cell_data

    return results


def extract_nuclear_cytoplasmic(
    intensity_image: np.ndarray,
    labeled_cells: np.ndarray,
    nuclear_channel: str = 'green',
    cytoplasmic_channel: str = 'red',
) -> List[Dict[str, float]]:
    """
    Extract nuclear and cytoplasmic fluorescence for each cell.

    Args:
        intensity_image: RGB fluorescence image
        labeled_cells: Labeled cell image from segmentation
        nuclear_channel: Color channel for nuclear fluorescence (default: 'green')
        cytoplasmic_channel: Color channel for cytoplasmic fluorescence (default: 'red')

    Returns:
        List of dicts with 'nuclear' and 'cytoplasmic' mean intensities per cell
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}

    cell_ids = np.unique(labeled_cells)
    cell_ids = cell_ids[cell_ids > 0]

    results = []

    for cell_id in cell_ids:
        cell_mask = labeled_cells == cell_id

        nuclear_idx = channel_map[nuclear_channel.lower()]
        nuclear_image = intensity_image[:, :, nuclear_idx]
        nuclear_intensity = float(np.mean(nuclear_image[cell_mask]))

        cyto_idx = channel_map[cytoplasmic_channel.lower()]
        cyto_image = intensity_image[:, :, cyto_idx]
        cyto_intensity = float(np.mean(cyto_image[cell_mask]))

        results.append({
            'nuclear': nuclear_intensity,
            'cytoplasmic': cyto_intensity,
        })

    return results


def track_cells_across_time(
    phase_images: List[np.ndarray],
    min_cell_area: int = 50,
) -> Tuple[Dict[int, List[Tuple[int, int, Tuple[float, float]]]], List[np.ndarray]]:
    """
    Segment (using nucleus/cell-body method) and track cell identities across time.

    Returns:
        Tuple of (tracks dict, list of labeled images per timepoint).
    """
    if len(phase_images) == 0:
        return {}, []

    labeled_images = []
    tracks = {}
    next_track_id = 0

    labeled = segment_cells(phase_images[0], min_cell_area)
    labeled_images.append(labeled)

    cell_ids = np.unique(labeled)
    cell_ids = cell_ids[cell_ids > 0]

    props = measure.regionprops(labeled)
    centroids = {prop.label: prop.centroid for prop in props}

    for track_id, cell_id in enumerate(cell_ids):
        centroid = centroids[cell_id]
        tracks[track_id] = [(0, int(cell_id), (float(centroid[0]), float(centroid[1])))]

    next_track_id = len(tracks)

    for t in range(1, len(phase_images)):
        curr_labels = segment_cells(phase_images[t], min_cell_area)
        labeled_images.append(curr_labels)

        prev_labels = labeled_images[t - 1]

        curr_cell_ids = np.unique(curr_labels)
        curr_cell_ids = curr_cell_ids[curr_cell_ids > 0]

        props = measure.regionprops(curr_labels)
        centroids = {prop.label: prop.centroid for prop in props}

        assigned = set()

        for curr_id in curr_cell_ids:
            curr_mask = curr_labels == curr_id

            overlaps = {}
            for prev_id in np.unique(prev_labels[curr_mask]):
                if prev_id == 0:
                    continue
                prev_mask = prev_labels == prev_id
                overlap = np.sum(curr_mask & prev_mask)
                overlaps[prev_id] = overlap

            if overlaps:
                best_prev_id = max(overlaps, key=overlaps.get)

                for track_id, track_list in tracks.items():
                    if any(entry[0] == t - 1 and entry[1] == best_prev_id for entry in track_list):
                        centroid = centroids[curr_id]
                        tracks[track_id].append((t, int(curr_id), (float(centroid[0]), float(centroid[1]))))
                        assigned.add(curr_id)
                        break
            else:
                centroid = centroids[curr_id]
                tracks[next_track_id] = [(t, int(curr_id), (float(centroid[0]), float(centroid[1])))]
                next_track_id += 1
                assigned.add(curr_id)

        for curr_id in curr_cell_ids:
            if curr_id not in assigned:
                centroid = centroids[curr_id]
                tracks[next_track_id] = [(t, int(curr_id), (float(centroid[0]), float(centroid[1])))]
                next_track_id += 1

    return tracks, labeled_images
