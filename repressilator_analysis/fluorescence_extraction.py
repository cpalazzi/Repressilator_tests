"""
Fluorescence extraction utilities for Repressilator analysis.

This module provides functions to segment cells from phase contrast images
and extract fluorescence intensity values for each cell and protein.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from skimage import feature, filters, measure, morphology, segmentation
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def _label_count(labeled: np.ndarray) -> int:
    labels = np.unique(labeled)
    return int(np.sum(labels > 0))


def _split_region_by_principal_axis(region_mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    coords = np.column_stack(np.nonzero(region_mask))
    if len(coords) < 2 * min_area:
        return []

    centered = coords - np.mean(coords, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    projection = centered @ major_axis
    cut_value = np.median(projection)

    left_coords = coords[projection <= cut_value]
    right_coords = coords[projection > cut_value]
    if len(left_coords) < min_area or len(right_coords) < min_area:
        return []

    left_mask = np.zeros_like(region_mask, dtype=bool)
    right_mask = np.zeros_like(region_mask, dtype=bool)
    left_mask[left_coords[:, 0], left_coords[:, 1]] = True
    right_mask[right_coords[:, 0], right_coords[:, 1]] = True
    return [left_mask, right_mask]


def _compact_labels(labeled: np.ndarray) -> np.ndarray:
    output = np.zeros_like(labeled, dtype=np.int32)
    labels = np.unique(labeled)
    labels = labels[labels > 0]
    for new_label, old_label in enumerate(labels, start=1):
        output[labeled == old_label] = new_label
    return output


def _estimate_nuclear_centroids(
    labeled_cells: np.ndarray,
    intensity_image: Optional[np.ndarray],
    min_cell_area: int,
) -> Dict[int, Tuple[float, float]]:
    if intensity_image is None:
        return {}

    intensity_gray = intensity_image[:, :, 1] if intensity_image.ndim == 3 else intensity_image
    smoothed = filters.gaussian(intensity_gray, sigma=1.0)
    thresholds = filters.threshold_multiotsu(smoothed, classes=3)
    intensity_classes = np.digitize(smoothed, bins=thresholds)
    nucleus_mask = intensity_classes == 2

    min_nucleus_area = max(3, min_cell_area // 2)
    nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=min_nucleus_area)

    nucleus_centroids = {}
    cell_ids = np.unique(labeled_cells)
    cell_ids = cell_ids[cell_ids > 0]

    for cell_id in cell_ids:
        cell_mask = labeled_cells == cell_id
        cell_nucleus = nucleus_mask & cell_mask

        # If the global nucleus class misses this cell, use local bright pixels.
        if np.sum(cell_nucleus) < min_nucleus_area:
            local_values = smoothed[cell_mask]
            if local_values.size > 0:
                local_threshold = np.quantile(local_values, 0.98)
                cell_nucleus = cell_mask & (smoothed >= local_threshold)

        if not np.any(cell_nucleus):
            continue

        ys, xs = np.nonzero(cell_nucleus)
        nucleus_centroids[int(cell_id)] = (float(np.mean(ys)), float(np.mean(xs)))

    return nucleus_centroids


def _extract_tracking_features(
    labeled_cells: np.ndarray,
    intensity_image: Optional[np.ndarray],
    min_cell_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    props = measure.regionprops(labeled_cells)
    if len(props) == 0:
        empty = np.array([], dtype=float)
        return (
            np.array([], dtype=np.int32),
            np.empty((0, 2), dtype=float),
            empty,
            empty,
            np.empty((0, 2), dtype=float),
        )

    labels = np.array([prop.label for prop in props], dtype=np.int32)
    centroids = np.array([prop.centroid for prop in props], dtype=float)
    areas = np.array([prop.area for prop in props], dtype=float)
    eccentricities = np.array([prop.eccentricity for prop in props], dtype=float)

    nucleus_map = _estimate_nuclear_centroids(labeled_cells, intensity_image, min_cell_area)
    nucleus_centroids = np.array(
        [nucleus_map.get(int(label), (float(centroids[i, 0]), float(centroids[i, 1]))) for i, label in enumerate(labels)],
        dtype=float,
    )

    return labels, centroids, areas, eccentricities, nucleus_centroids


def _overlap_matrix(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    prev_ids: np.ndarray,
    curr_ids: np.ndarray,
) -> np.ndarray:
    overlap = np.zeros((len(prev_ids), len(curr_ids)), dtype=float)
    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return overlap

    curr_lookup = {int(label): idx for idx, label in enumerate(curr_ids)}
    for i, prev_id in enumerate(prev_ids):
        prev_mask = prev_labels == prev_id
        overlapping_labels, counts = np.unique(curr_labels[prev_mask], return_counts=True)
        for label, count in zip(overlapping_labels, counts):
            if label == 0:
                continue
            j = curr_lookup.get(int(label))
            if j is not None:
                overlap[i, j] = float(count)

    return overlap


def _match_cells_between_frames(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    prev_intensity: Optional[np.ndarray],
    curr_intensity: Optional[np.ndarray],
    min_cell_area: int,
    predicted_prev_centroids: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev_ids, prev_centroids, prev_areas, prev_ecc, prev_nuc = _extract_tracking_features(
        prev_labels,
        prev_intensity,
        min_cell_area,
    )
    curr_ids, curr_centroids, curr_areas, curr_ecc, curr_nuc = _extract_tracking_features(
        curr_labels,
        curr_intensity,
        min_cell_area,
    )

    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return prev_ids, curr_ids, np.array([], dtype=int), np.array([], dtype=int)

    overlap = _overlap_matrix(prev_labels, curr_labels, prev_ids, curr_ids)
    union = prev_areas[:, np.newaxis] + curr_areas[np.newaxis, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    overlap_cost = 1.0 - iou

    centroid_dist = np.linalg.norm(prev_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :], axis=2)
    nearest_centroid = np.min(centroid_dist, axis=1)
    centroid_scale = max(float(np.percentile(nearest_centroid, 90)), 4.0)
    centroid_cost = np.clip(centroid_dist / centroid_scale, 0.0, 4.0)

    if predicted_prev_centroids is not None:
        predicted_centroids = np.array(
            [
                predicted_prev_centroids.get(
                    int(label),
                    (float(prev_centroids[i, 0]), float(prev_centroids[i, 1])),
                )
                for i, label in enumerate(prev_ids)
            ],
            dtype=float,
        )
    else:
        predicted_centroids = prev_centroids

    predicted_dist = np.linalg.norm(predicted_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :], axis=2)
    nearest_predicted = np.min(predicted_dist, axis=1)
    predicted_scale = max(float(np.percentile(nearest_predicted, 90)), 4.0)
    predicted_cost = np.clip(predicted_dist / predicted_scale, 0.0, 4.0)

    nucleus_dist = np.linalg.norm(prev_nuc[:, np.newaxis, :] - curr_nuc[np.newaxis, :, :], axis=2)
    nearest_nucleus = np.min(nucleus_dist, axis=1)
    nucleus_scale = max(float(np.percentile(nearest_nucleus, 90)), 3.0)
    nucleus_cost = np.clip(nucleus_dist / nucleus_scale, 0.0, 4.0)

    area_ratio = np.divide(curr_areas[np.newaxis, :], np.maximum(prev_areas[:, np.newaxis], 1.0))
    area_cost = np.abs(np.log(np.clip(area_ratio, 1e-3, 1e3)))
    area_cost = np.clip(area_cost, 0.0, 4.0)

    shape_cost = np.abs(prev_ecc[:, np.newaxis] - curr_ecc[np.newaxis, :])

    # Combined cost: overlap dominates, centroid/nucleus resolve overlaps, area/shape stabilize.
    cost_matrix = (
        0.38 * overlap_cost
        + 0.25 * predicted_cost
        + 0.18 * centroid_cost
        + 0.12 * nucleus_cost
        + 0.05 * area_cost
        + 0.02 * shape_cost
    )
    cost_matrix += (overlap <= 0) * 0.35

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return prev_ids, curr_ids, row_ind.astype(int), col_ind.astype(int)


def _enforce_expected_cell_count(
    labeled: np.ndarray,
    expected_cells: Optional[int],
    min_cell_area: int,
) -> np.ndarray:
    if expected_cells is None:
        return labeled

    current = _label_count(labeled)
    if current == expected_cells:
        return labeled

    adjusted = labeled.copy()

    # Merge tiny over-segmented fragments by dropping smallest labels first.
    while current > expected_cells:
        props = measure.regionprops(adjusted)
        if len(props) == 0:
            break
        smallest = min(props, key=lambda p: p.area)
        adjusted[adjusted == smallest.label] = 0
        current = _label_count(adjusted)

    # Split largest merged labels until target count is reached.
    while current < expected_cells:
        prev_current = current
        required_increase = expected_cells - current
        props = measure.regionprops(adjusted)
        if len(props) == 0:
            break

        largest = max(props, key=lambda p: p.area)
        region_mask = adjusted == largest.label
        distance = ndimage.distance_transform_edt(region_mask)
        peaks = feature.peak_local_max(distance, labels=region_mask, min_distance=3, num_peaks=2)
        valid_submasks = []
        strict_min_area = max(min_cell_area * 2, 20)

        if len(peaks) >= 2:
            local_markers = np.zeros_like(distance, dtype=np.int32)
            local_markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
            local_markers = ndimage.label(local_markers > 0)[0]
            split_local = segmentation.watershed(-distance, local_markers, mask=region_mask)

            split_ids = np.unique(split_local)
            split_ids = split_ids[split_ids > 0]

            for split_id in split_ids:
                submask = split_local == split_id
                if np.sum(submask) < strict_min_area:
                    continue
                valid_submasks.append(submask)

        # Need at least two valid parts to perform an actual split. If watershed
        # cannot split, use a deterministic geometric bisect on the largest region.
        if len(valid_submasks) < 2:
            fallback_submasks = _split_region_by_principal_axis(
                region_mask,
                min_area=max(min_cell_area, 10),
            )
            if len(fallback_submasks) < 2:
                break
            valid_submasks = fallback_submasks

        # Prevent overshooting the expected count.
        max_parts = required_increase + 1
        if len(valid_submasks) > max_parts:
            valid_submasks = sorted(valid_submasks, key=lambda m: int(np.sum(m)), reverse=True)[:max_parts]

        # Remove old label and insert split labels with new ids
        adjusted[region_mask] = 0
        max_label = int(np.max(adjusted))
        for submask in valid_submasks:
            max_label += 1
            adjusted[submask] = max_label

        current = _label_count(adjusted)

        # Stop if we cannot make progress or reached the target.
        if current <= prev_current or current >= expected_cells:
            break

    return _compact_labels(adjusted)


def segment_cells(
    phase_image: np.ndarray,
    min_cell_area: int = 50,
    intensity_image: Optional[np.ndarray] = None,
) -> np.ndarray:
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

    # Smooth noise before thresholding
    smoothed = filters.gaussian(gray, sigma=1.0)

    # Multi-Otsu (3 classes): keep the darkest class as foreground.
    # In these phase images, this isolates cell interiors and reduces merges.
    thresholds = filters.threshold_multiotsu(smoothed, classes=3)
    intensity_classes = np.digitize(smoothed, bins=thresholds)
    core_binary = intensity_classes == 0
    binary = intensity_classes <= 1

    # Clean up binary image
    core_binary = morphology.remove_small_objects(core_binary, min_size=max(3, min_cell_area))
    binary = morphology.remove_small_objects(binary, min_size=min_cell_area)
    binary = morphology.remove_small_holes(binary, area_threshold=min_cell_area)
    core_binary = core_binary & binary

    # Split touching cells via watershed on distance transform
    seed_mask = core_binary if np.any(core_binary) else binary
    distance = ndimage.distance_transform_edt(seed_mask)

    # Geometric seeds from distance transform
    dist_peaks = feature.peak_local_max(distance, labels=seed_mask, min_distance=4)

    # Optional fluorescence-guided seeds from nuclei.
    peaks = dist_peaks
    if intensity_image is not None:
        intensity_gray = intensity_image[:, :, 1] if intensity_image.ndim == 3 else intensity_image
        marker_source = filters.gaussian(intensity_gray, sigma=1.0)

        # Build a nuclear mask from the brightest fluorescence class.
        nucleus_thresholds = filters.threshold_multiotsu(marker_source, classes=3)
        nucleus_classes = np.digitize(marker_source, bins=nucleus_thresholds)
        nucleus_mask = (nucleus_classes == 2) & binary

        min_nucleus_area = max(3, min_cell_area // 2)
        nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size=min_nucleus_area)
        nucleus_mask = morphology.remove_small_holes(nucleus_mask, area_threshold=min_nucleus_area)

        nucleus_labels = measure.label(nucleus_mask)
        refined_nuclei = np.zeros_like(nucleus_labels, dtype=np.int32)
        next_label = 1
        split_seed_labels = set()

        for region in measure.regionprops(nucleus_labels):
            region_mask = nucleus_labels == region.label

            # Elongated nuclei are likely merged; split them before cell declumping.
            if region.eccentricity >= 0.9 and region.area >= 2 * min_nucleus_area:
                nuc_distance = ndimage.distance_transform_edt(region_mask)
                nuc_peaks = feature.peak_local_max(
                    nuc_distance,
                    labels=region_mask,
                    min_distance=2,
                )

                if len(nuc_peaks) >= 2:
                    local_markers = np.zeros_like(nuc_distance, dtype=np.int32)
                    local_markers[tuple(nuc_peaks.T)] = np.arange(1, len(nuc_peaks) + 1, dtype=np.int32)
                    local_markers = ndimage.label(local_markers > 0)[0]
                    split_nuclei = segmentation.watershed(-nuc_distance, local_markers, mask=region_mask)

                    split_ids = np.unique(split_nuclei)
                    split_ids = split_ids[split_ids > 0]
                    wrote_split = False

                    for split_id in split_ids:
                        split_mask = split_nuclei == split_id
                        if np.sum(split_mask) < min_nucleus_area:
                            continue
                        refined_nuclei[split_mask] = next_label
                        split_seed_labels.add(next_label)
                        next_label += 1
                        wrote_split = True

                    if wrote_split:
                        continue

            refined_nuclei[region_mask] = next_label
            next_label += 1

        nucleus_peaks = []
        for prop in measure.regionprops(refined_nuclei):
            if prop.label not in split_seed_labels:
                continue
            y = int(np.clip(round(prop.centroid[0]), 0, binary.shape[0] - 1))
            x = int(np.clip(round(prop.centroid[1]), 0, binary.shape[1] - 1))
            nucleus_peaks.append([y, x])

        if len(nucleus_peaks) > 0:
            nucleus_peaks = np.asarray(nucleus_peaks, dtype=int)
            peaks = nucleus_peaks if len(peaks) == 0 else np.vstack([peaks, nucleus_peaks])

    if len(peaks) > 0:
        peaks = np.unique(peaks, axis=0)

    markers = np.zeros_like(distance, dtype=np.int32)
    if len(peaks) > 0:
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
    markers = ndimage.label(markers > 0)[0]

    if np.max(markers) > 0:
        labeled = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labeled = measure.label(binary)

    # Clear border objects (cells touching image edge)
    labeled = segmentation.clear_border(labeled)

    # Remove very small fragments produced by watershed
    labeled = morphology.remove_small_objects(labeled, min_size=max(min_cell_area * 4, 20))

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
    intensity_images: Optional[List[np.ndarray]] = None,
    expected_cells: Optional[int] = 80,
) -> Tuple[Dict[int, List[Tuple[int, int, Tuple[float, float]]]], List[np.ndarray]]:
    """
    Segment and track cell identities across time points.

    This function segments cells from phase contrast images and tracks them
    across frames with one-to-one assignment. Matching combines mask overlap,
    centroid/nucleus motion, and mild area/shape continuity constraints.

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

    # Process first frame
    first_intensity = intensity_images[0] if intensity_images is not None else None
    first_labels = segment_cells(phase_images[0], min_cell_area, intensity_image=first_intensity)
    first_labels = _enforce_expected_cell_count(first_labels, expected_cells, min_cell_area)
    labeled_images.append(first_labels)

    first_props = measure.regionprops(first_labels)
    first_cell_ids = sorted([int(prop.label) for prop in first_props])
    first_centroids = {int(prop.label): prop.centroid for prop in first_props}

    label_to_track = {}
    for track_id, cell_id in enumerate(first_cell_ids):
        centroid = first_centroids[cell_id]
        tracks[track_id] = [(0, cell_id, (float(centroid[0]), float(centroid[1])))]
        label_to_track[cell_id] = track_id

    next_track_id = len(tracks)

    # Process subsequent frames
    for t in range(1, len(phase_images)):
        prev_labels = labeled_images[t - 1]
        prev_intensity = intensity_images[t - 1] if intensity_images is not None else None

        current_intensity = intensity_images[t] if intensity_images is not None else None
        curr_labels = segment_cells(phase_images[t], min_cell_area, intensity_image=current_intensity)
        curr_labels = _enforce_expected_cell_count(curr_labels, expected_cells, min_cell_area)
        labeled_images.append(curr_labels)

        curr_props = measure.regionprops(curr_labels)
        curr_centroids = {int(prop.label): prop.centroid for prop in curr_props}
        curr_cell_ids = sorted([int(prop.label) for prop in curr_props])

        predicted_prev_centroids = {}
        for prev_label, track_id in label_to_track.items():
            history = tracks.get(track_id, [])
            if len(history) >= 2:
                y1, x1 = history[-1][2]
                y0, x0 = history[-2][2]
                predicted_prev_centroids[int(prev_label)] = (2.0 * y1 - y0, 2.0 * x1 - x0)
            elif len(history) == 1:
                predicted_prev_centroids[int(prev_label)] = history[-1][2]

        prev_ids, matched_curr_ids, row_ind, col_ind = _match_cells_between_frames(
            prev_labels,
            curr_labels,
            prev_intensity,
            current_intensity,
            min_cell_area,
            predicted_prev_centroids=predicted_prev_centroids,
        )

        assigned_curr = set()
        next_label_to_track = {}

        for prev_idx, curr_idx in zip(row_ind, col_ind):
            prev_label = int(prev_ids[prev_idx])
            curr_label = int(matched_curr_ids[curr_idx])

            track_id = label_to_track.get(prev_label)
            if track_id is None:
                continue

            centroid = curr_centroids[curr_label]
            tracks[track_id].append((t, curr_label, (float(centroid[0]), float(centroid[1]))))
            next_label_to_track[curr_label] = track_id
            assigned_curr.add(curr_label)

        # Create new tracks only for truly unmatched labels (for variable-population data).
        for curr_label in curr_cell_ids:
            if curr_label in assigned_curr:
                continue
            centroid = curr_centroids[curr_label]
            tracks[next_track_id] = [(t, curr_label, (float(centroid[0]), float(centroid[1])))]
            next_label_to_track[curr_label] = next_track_id
            next_track_id += 1

        label_to_track = next_label_to_track

    return tracks, labeled_images
