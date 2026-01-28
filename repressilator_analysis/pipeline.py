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






def extract_protein_numbers_from_tracks(
    tracks,
    intensity_dir: str,
    phase_dir: str,
    calibration_files: List[str],
    weights: List[float],
    track_labels: List[str] = None,
    calibration_headers: int = 1,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Extract protein molecule numbers from tracked cells across timepoints.

    Args:
        tracks: Tuple of (tracks_dict, labeled_images) from track_cells_across_time.
                tracks_dict maps track_id -> [(timepoint_idx, cell_label, (y, x)), ...]
                labeled_images is a list of labeled cell images (one per timepoint)
        intensity_dir: Directory with fluorescence intensity images
        phase_dir: Directory with phase contrast images
        calibration_files: List of calibration file paths for each protein
        weights: List of molecular weights (kDa) for each protein
        track_labels: Labels for the proteins (e.g., ["n_intensity", "c_intensity"])
        calibration_headers: Number of header rows to skip in calibration files
        output_dir: Optional directory to save results

    Returns:
        3D numpy array with shape (timepoints, cells, proteins) containing molecule counts
    """
    # Unpack tracks tuple
    tracks_dict, labeled_images = tracks

    # Load images
    timepoints, intensity_images, phase_images = image_loader.load_timeseries(
        intensity_dir, phase_dir
    )

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