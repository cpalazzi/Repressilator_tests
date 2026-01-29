"""
Fluorescence extraction utilities for Repressilator analysis.

This module provides functions to segment cells from phase contrast images
and extract fluorescence intensity values for each cell and protein.
"""

import numpy as np
from typing import List, Dict, Tuple
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage


def segment_cells(phase_image: np.ndarray, min_cell_area: int = 50) -> np.ndarray:
    """
    Segment individual cells from a phase contrast image.

    Args:
        phase_image: Phase contrast image (grayscale or RGB)
        min_cell_area: Minimum cell area in pixels

    Returns:
        Labeled image where each cell has a unique integer label
    """
    # Convert to grayscale if RGB
    if phase_image.ndim == 3:
        gray = np.mean(phase_image, axis=2)
    else:
        gray = phase_image

    # Apply Otsu's thresholding
    threshold = filters.threshold_otsu(gray)
    binary = gray < threshold  # Cells are typically darker in phase contrast

    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=min_cell_area)
    binary = morphology.remove_small_holes(binary, area_threshold=min_cell_area)

    # Label connected components
    labeled = measure.label(binary)

    # Clear border objects (cells touching image edge)
    labeled = segmentation.clear_border(labeled)

    return labeled


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

    # Get unique cell labels (excluding background = 0)
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

            # Extract mean fluorescence in this cell
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

    This function extracts mean fluorescence intensities from two separate
    channels representing nuclear and cytoplasmic compartments.

    Args:
        intensity_image: RGB fluorescence image
        labeled_cells: Labeled cell image from segmentation
        nuclear_channel: Color channel for nuclear fluorescence (default: 'green')
        cytoplasmic_channel: Color channel for cytoplasmic fluorescence (default: 'red')

    Returns:
        List of dictionaries with keys 'nuclear' and 'cytoplasmic' containing
        mean fluorescence intensities for each cell
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}

    cell_ids = np.unique(labeled_cells)
    cell_ids = cell_ids[cell_ids > 0]

    results = []

    for cell_id in cell_ids:
        cell_mask = labeled_cells == cell_id

        # Extract nuclear fluorescence
        nuclear_idx = channel_map[nuclear_channel.lower()]
        nuclear_image = intensity_image[:, :, nuclear_idx]
        nuclear_intensity = float(np.mean(nuclear_image[cell_mask]))

        # Extract cytoplasmic fluorescence
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
    Segment and track cell identities across time points based on spatial overlap.

    This function segments cells from phase contrast images and tracks them
    across frames using maximum overlap between consecutive frames.

    Args:
        phase_images: List of phase contrast images (one per timepoint)
        min_cell_area: Minimum cell area in pixels for segmentation

    Returns:
        Tuple of:
        - Dictionary mapping track_id -> [(timepoint_idx, cell_label, (y, x)), ...]
          where (y, x) is the centroid pixel location of the cell
        - List of labeled cell images (one per timepoint)
    """
    if len(phase_images) == 0:
        return {}, []

    labeled_images = []
    tracks = {}
    next_track_id = 0

    # Process first frame
    labeled = segment_cells(phase_images[0], min_cell_area)
    labeled_images.append(labeled)

    # Initialize tracks with cells from first frame
    cell_ids = np.unique(labeled)
    cell_ids = cell_ids[cell_ids > 0]

    # Get centroids for first frame
    props = measure.regionprops(labeled)
    centroids = {prop.label: prop.centroid for prop in props}

    for track_id, cell_id in enumerate(cell_ids):
        centroid = centroids[cell_id]
        tracks[track_id] = [(0, int(cell_id), (float(centroid[0]), float(centroid[1])))]

    next_track_id = len(tracks)

    # Process subsequent frames
    for t in range(1, len(phase_images)):
        # Segment current frame
        curr_labels = segment_cells(phase_images[t], min_cell_area)
        labeled_images.append(curr_labels)

        prev_labels = labeled_images[t - 1]

        curr_cell_ids = np.unique(curr_labels)
        curr_cell_ids = curr_cell_ids[curr_cell_ids > 0]

        # Get centroids for current frame
        props = measure.regionprops(curr_labels)
        centroids = {prop.label: prop.centroid for prop in props}

        assigned = set()

        # For each current cell, find best match in previous frame
        for curr_id in curr_cell_ids:
            curr_mask = curr_labels == curr_id

            # Find overlap with previous frame cells
            overlaps = {}
            for prev_id in np.unique(prev_labels[curr_mask]):
                if prev_id == 0:
                    continue
                prev_mask = prev_labels == prev_id
                overlap = np.sum(curr_mask & prev_mask)
                overlaps[prev_id] = overlap

            # Assign to track with maximum overlap
            if overlaps:
                best_prev_id = max(overlaps, key=overlaps.get)

                # Find which track this previous cell belongs to
                for track_id, track_list in tracks.items():
                    if any(entry[0] == t - 1 and entry[1] == best_prev_id for entry in track_list):
                        centroid = centroids[curr_id]
                        tracks[track_id].append((t, int(curr_id), (float(centroid[0]), float(centroid[1]))))
                        assigned.add(curr_id)
                        break
            else:
                # New cell appeared
                centroid = centroids[curr_id]
                tracks[next_track_id] = [(t, int(curr_id), (float(centroid[0]), float(centroid[1])))]
                next_track_id += 1
                assigned.add(curr_id)

        # Handle unassigned cells (new cells)
        for curr_id in curr_cell_ids:
            if curr_id not in assigned:
                centroid = centroids[curr_id]
                tracks[next_track_id] = [(t, int(curr_id), (float(centroid[0]), float(centroid[1])))]
                next_track_id += 1

    return tracks, labeled_images
