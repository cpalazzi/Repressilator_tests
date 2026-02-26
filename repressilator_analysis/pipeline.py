"""
Main analysis pipeline for Repressilator fluorescence microscopy data.

This script orchestrates the complete analysis workflow:
1. Load time-series images
2. Segment cells and extract fluorescence
3. Convert to protein quantities
4. Infer ODE parameters using PINTS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import pickle

from . import image_loader
from . import fluorescence_extraction
from . import calibration
from . import ode_inference






def full_analysis(
    intensity_dir: str,
    phase_dir: str,
    calibration_dict: Dict,
    cell_ids: Optional[List[int]] = None,
    min_cell_area: int = 5,
    method: str = 'differential_evolution',
) -> tuple:
    """
    Run the complete Repressilator analysis pipeline from images to parameters.

    This function orchestrates the entire analysis workflow:
    1. Loads time-series fluorescence and phase contrast images
    2. Segments and tracks cells across all timepoints
    3. Extracts fluorescence intensities and converts to molecule counts
    4. Infers Repressilator ODE parameters for specified cells

    Args:
        intensity_dir: Directory containing fluorescence intensity images
        phase_dir: Directory containing phase contrast images
        calibration_dict: Dictionary with keys:
            - "files": List of calibration file paths (one per protein/channel)
            - "weights": List of molecular weights in kDa (one per protein)
            - "headers": Number of header rows to skip in calibration files
        cell_ids: Optional list of specific cell IDs to analyze (defaults to all tracked cells)
        min_cell_area: Minimum cell area in pixels for segmentation (default: 5)
        method: Optimization method for parameter inference ('differential_evolution' or 'least_squares')

    Returns:
        Tuple of (tracks, protein_numbers, parameters) where:
        - tracks: List of dicts per timepoint, each dict has 'cell_id' and 'centre' [y, x]
        - protein_numbers: 3D numpy array of shape (timepoints, cells, proteins) with molecule counts
        - parameters: Dict mapping cell_id -> array of best-fit parameters
          [alpha, alpha0, beta, hill, mrna_half_life, p_half_life]
    """
    # Load images
    timepoints, intensity_images, phase_images = image_loader.load_timeseries(
        intensity_dir, phase_dir
    )

    # Track cells
    tracks_dict, labeled_images = fluorescence_extraction.track_cells_across_time(
        phase_images,
        min_cell_area,
        intensity_images=intensity_images,
    )

    # Convert tracks to list format for test compatibility
    tracks_list = [[] for _ in range(len(phase_images))]
    for track_id, track_data in tracks_dict.items():
        for tp_idx, cell_label, centroid in track_data:
            tracks_list[tp_idx].append({
                "cell_id": int(track_id),
                "centre": list(centroid)
            })

    # Extract protein numbers
    protein_numbers = extract_protein_numbers_from_tracks(
        (tracks_dict, labeled_images),
        timepoints,
        intensity_images,
        phase_images,
        calibration_dict=calibration_dict,
        track_labels=["n_intensity", "c_intensity"],
    )

    # Run ODE inference for specified cells
    parameters = {}

    # If no cell_ids specified, analyze all cells
    if cell_ids is None:
        cell_ids = list(tracks_dict.keys())

    for cell_id in cell_ids:
        print(f"\nInferring parameters for cell {cell_id}...")

        # Extract time series for this cell
        cell_observations = protein_numbers[:, cell_id, :]

        # Check for NaN values
        if np.any(np.isnan(cell_observations)):
            print(f"Warning: Cell {cell_id} has missing data, skipping...")
            continue

        # Run inference
        best_params = ode_inference.infer_parameters(
            timepoints,
            cell_observations,
            method=method
        )

        parameters[cell_id] = best_params

    return tracks_list, protein_numbers, parameters


def extract_protein_numbers_from_tracks(
    tracks,
    timepoints: np.ndarray,
    intensity_images: List[np.ndarray],
    phase_images: List[np.ndarray],
    calibration_dict: Dict = None,
    calibration_files: List[str] = None,
    weights: List[float] = None,
    track_labels: List[str] = None,
    calibration_headers: int = 1,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Extract protein molecule numbers from tracked cells across all timepoints.

    This function takes cell tracking data and fluorescence images, extracts
    mean pixel intensities for each cell and channel, then converts to molecule
    counts using calibration curves.

    Args:
        tracks: Tuple of (tracks_dict, labeled_images) from track_cells_across_time where:
                - tracks_dict maps track_id -> [(timepoint_idx, cell_label, (y, x)), ...]
                - labeled_images is a list of labeled cell segmentation masks (one per timepoint)
        timepoints: Array of timepoints in minutes
        intensity_images: List of fluorescence intensity image arrays
        phase_images: List of phase contrast image arrays
        calibration_dict: Dictionary with keys "files", "weights", and "headers" for calibration.
                          If provided, overrides individual calibration parameters.
        calibration_files: List of calibration file paths (one per protein/channel).
                          Ignored if calibration_dict is provided.
        weights: List of molecular weights in kDa (one per protein, same order as calibration_files).
                Ignored if calibration_dict is provided.
        track_labels: Optional labels for proteins (e.g., ["n_intensity", "c_intensity"])
        calibration_headers: Number of header rows to skip in calibration files (default: 1).
                            Ignored if calibration_dict is provided.
        output_dir: Optional directory path to save results (currently unused)

    Returns:
        3D numpy array with shape (n_timepoints, n_cells, n_proteins) containing
        molecule counts for each cell at each timepoint. Missing/untracked cells
        have NaN values.
    """
    # Handle calibration dictionary or individual parameters
    if calibration_dict is not None:
        calibration_files = calibration_dict["files"]
        weights = calibration_dict["weights"]
        calibration_headers = calibration_dict["headers"]
    elif calibration_files is None or weights is None:
        raise ValueError("Must provide either calibration_dict or both calibration_files and weights")

    # Unpack tracks tuple
    tracks_dict, labeled_images = tracks

    n_timepoints = len(timepoints)
    n_cells = len(tracks_dict)
    n_proteins = len(calibration_files)

    # Initialize output array
    protein_numbers = np.full((n_timepoints, n_cells, n_proteins), np.nan)

    # Load calibrations
    calibrations = []
    for cal_file, weight in zip(calibration_files, weights):
        cal = calibration.ProteinCalibration(cal_file, weight, header=calibration_headers)
        calibrations.append(cal)

    # Build a mapping from (timepoint, cell_label) to track_id
    timepoint_cell_to_track = {}
    for track_id, track_data in tracks_dict.items():
        for tp_idx, cell_label, centroid in track_data:
            timepoint_cell_to_track[(tp_idx, cell_label)] = track_id

    # Process each timepoint
    for t_idx, (intensity_img, labeled_img) in enumerate(zip(intensity_images, labeled_images)):
        # Get unique cell IDs in this frame
        cell_ids = np.unique(labeled_img)
        cell_ids = cell_ids[cell_ids > 0]

        # Extract fluorescence for each cell
        for cell_id in cell_ids:
            # Get track_id for this cell
            if (t_idx, int(cell_id)) not in timepoint_cell_to_track:
                continue
            track_id = timepoint_cell_to_track[(t_idx, int(cell_id))]

            # Get cell mask
            cell_mask = labeled_img == cell_id

            # Extract fluorescence for each protein (channel)
            # Assuming nuclear protein is in green channel (channel 1)
            # and cytoplasmic protein is in red channel (channel 0)
            if intensity_img.ndim == 3:
                # Multi-channel image
                nuclear_pixels = intensity_img[:, :, 1][cell_mask]  # Green channel
                cytoplasmic_pixels = intensity_img[:, :, 0][cell_mask]  # Red channel
            else:
                # Single channel - use same for both
                pixels = intensity_img[cell_mask]
                nuclear_pixels = pixels
                cytoplasmic_pixels = pixels

            # Calculate mean intensity for each protein
            intensities = [
                np.mean(nuclear_pixels),
                np.mean(cytoplasmic_pixels)
            ]

            # Convert to molecule counts using calibrations
            for protein_idx, (intensity, cal) in enumerate(zip(intensities, calibrations)):
                molecules = cal.pixel_intensities_to_molecules(intensity)
                protein_numbers[t_idx, track_id, protein_idx] = molecules

    return protein_numbers