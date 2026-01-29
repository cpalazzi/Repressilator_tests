# Repressilator Analysis

A Python package for analyzing time-series fluorescence microscopy images of bacterial cells expressing the Repressilator genetic circuit.

## Overview

The Repressilator is a synthetic genetic regulatory network consisting of three transcription repressors that form a cyclic negative feedback loop (Elowitz & Leibler, 2000). This package provides a complete analysis pipeline for quantifying protein dynamics from fluorescence microscopy:

1. **Image loading**: Read and organize time-series fluorescence and phase contrast images
2. **Cell segmentation**: Identify individual bacterial cells using phase contrast images
3. **Cell tracking**: Track cells across timepoints using spatial overlap
4. **Fluorescence extraction**: Measure fluorescence intensities for each protein in each cell
5. **Calibration**: Convert pixel intensities to absolute protein molecule counts
6. **ODE parameter inference**: Fit Repressilator model parameters using optimization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ClaudeRepressilator

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Dependencies
- numpy
- scipy
- scikit-image
- matplotlib
- pandas
- pints (for ODE modeling)

## Package Structure

```
repressilator_analysis/
├── __init__.py              # Package initialization
├── image_loader.py          # Load and sort time-series images
├── fluorescence_extraction.py  # Cell segmentation, tracking, and fluorescence extraction
├── calibration.py           # Convert pixel intensities to molecule counts
├── ode_inference.py         # ODE model and parameter inference
├── pipeline.py              # Main analysis pipeline
├── utils.py                 # Utility functions
└── __main__.py              # Command-line entry point
```

## Quick Start

### Using the Full Pipeline

```python
from repressilator_analysis import pipeline
import os

# Define calibration files
calibration_dict = {
    "files": [
        "docs/Nuclear repressor 1 (66 kDa) calibration.txt",
        "docs/Cytosolic repressor (53 kDa) calibration.txt"
    ],
    "weights": [66, 53],  # Molecular weights in kDa
    "headers": 1          # Number of header rows
}

# Run the full analysis
tracks, protein_numbers, parameters = pipeline.full_analysis(
    intensity_dir="images/intensity",
    phase_dir="images/phase",
    calibration_dict=calibration_dict,
    cell_ids=None,  # Analyze all cells (or specify a list of cell IDs)
    min_cell_area=5,
    method='differential_evolution'
)

# tracks: List of cells per timepoint with positions
# protein_numbers: 3D array (timepoints × cells × proteins) with molecule counts
# parameters: Dict mapping cell_id → fitted ODE parameters
```

### Using Individual Modules

```python
from repressilator_analysis import image_loader, fluorescence_extraction, calibration

# Load images
timepoints, intensity_images, phase_images = image_loader.load_timeseries(
    "images/intensity",
    "images/phase"
)

# Track cells across time
tracks_dict, labeled_images = fluorescence_extraction.track_cells_across_time(
    phase_images,
    min_cell_area=5
)

# Load calibration
cal_nuclear = calibration.ProteinCalibration(
    "docs/Nuclear repressor 1 (66 kDa) calibration.txt",
    molecular_weight_kda=66,
    header=1
)

# Convert pixel intensities to molecule counts
pixel_intensity = 100.0
molecules = cal_nuclear.pixel_intensities_to_molecules(pixel_intensity)
print(f"Pixel intensity {pixel_intensity} = {molecules:.2e} molecules")
```

## Data Format

### Input Images

**Fluorescence Intensity Images** (`images/intensity/`)
- RGB PNG format (512×512 pixels typical)
- Naming convention: `sample_t+{time}m.png` where time is in minutes
- Red channel: Cytosolic repressor fluorescence
- Green channel: Nuclear/nucleoid repressor fluorescence
- Blue channel: (unused)

**Phase Contrast Images** (`images/phase/`)
- Grayscale or RGB PNG format
- Used for cell segmentation
- Must correspond 1:1 with intensity images

### Calibration Data

Located in `docs/` directory:
- `Nuclear repressor 1 (66 kDa) calibration.txt`
- `Cytosolic repressor (53 kDa) calibration.txt`

Format:
```
Mass/ng    repeat 1/A.U.    repeat 2/A.U.    repeat 3/A.U.
2.500e+01  5.969e+02        6.228e+02        8.278e+02
...
```

**Conversion factor**: `PIXEL_TO_AU_FACTOR = 1e7` relates pixel intensities to calibration arbitrary units (A.U.)

## Key Features

### Cell Segmentation
- Uses Otsu thresholding on phase contrast images
- Morphological operations to clean up segmentation
- Removes cells touching image borders
- Configurable minimum cell area filter

### Cell Tracking
- Overlap-based tracking between consecutive frames
- Handles cell division and new cells appearing
- Tracks centroids for position validation
- Returns both track dictionary and labeled images

### Calibration System
- Linear interpolation between calibration points
- Extrapolation for values outside calibration range
- Automatic averaging of technical replicates
- Direct conversion: pixel intensity → fluorescence (A.U.) → mass (ng) → molecules

### ODE Model

The Repressilator model includes:
- 6 state variables: 3 mRNAs (m1, m2, m3) + 3 proteins (p1, p2, p3)
- Hill function repression with cyclic topology
- Observable outputs: 2 proteins (p1, p2)
- Unobserved protein: p3 (no fluorescence tag)

**Parameters** (6 total):
- `alpha`: Maximum transcription rate
- `alpha0`: Basal transcription rate
- `beta`: Translation rate
- `hill`: Hill coefficient (cooperativity)
- `mrna_half_life`: mRNA degradation half-life (minutes)
- `p_half_life`: Protein degradation half-life (minutes)

**Inference method**: Differential evolution (global optimization)
- Alternative: L-BFGS-B (local optimization)
- Fits parameters to minimize sum of squared residuals

## Output Structure

### Tracks
List of dictionaries per timepoint:
```python
tracks[timepoint_idx] = [
    {"cell_id": 0, "centre": [y, x]},
    {"cell_id": 1, "centre": [y, x]},
    ...
]
```

### Protein Numbers
3D numpy array with shape `(n_timepoints, n_cells, n_proteins)`:
```python
protein_numbers[t, c, p] = molecule_count
# t: timepoint index
# c: cell ID (track ID)
# p: protein index (0=nuclear, 1=cytoplasmic)
```

### Parameters
Dictionary mapping cell IDs to fitted parameter arrays:
```python
parameters[cell_id] = [alpha, alpha0, beta, hill, mrna_half_life, p_half_life]
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Tests include:
- Calibration accuracy validation
- Cell segmentation and tracking with ground truth positions
- Fluorescence extraction validation
- ODE model simulation accuracy
- Parameter inference on synthetic data

See `tests/README.md` for detailed testing documentation.

## References

- Elowitz, M. B., & Leibler, S. (2000). A synthetic oscillatory network of transcriptional regulators. *Nature*, 403(6767), 335-338. (See `docs/35002125.pdf`)

## License

[Specify license here]

## Contributing

[Contributing guidelines]
