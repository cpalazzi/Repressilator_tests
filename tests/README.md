# Tests Documentation

This directory contains the test suite for the Repressilator Analysis package. The tests validate all major components of the analysis pipeline against ground truth data.

## Test Structure

```
tests/
├── README.md                        # This file
├── calibration_test.py             # Calibration conversion tests
├── fluorescence_extractor_test.py  # Segmentation, tracking, and extraction tests
├── ode_inference_test.py           # ODE simulation and parameter inference tests
├── pipeline_test.py                # (Integration tests)
├── test_utils.py                   # Utility functions for testing
└── testdata/                       # Ground truth test data
    ├── F_vs_amount.txt             # Fluorescence and position ground truth
    └── protein_numbers/            # Synthetic ODE simulation data
        ├── simulation_001.txt      # Simulated protein time series
        └── parameters/
            └── params_001.json     # True parameters for simulation
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/calibration_test.py
pytest tests/fluorescence_extractor_test.py
pytest tests/ode_inference_test.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with print statements visible:
```bash
pytest tests/ -s
```

## Test Files

### 1. `calibration_test.py`

**Purpose**: Validate the calibration system for converting pixel intensities to protein molecule counts.

**Test**: `test_calibration()`
- Loads both protein calibration files (Nuclear 66 kDa, Cytosolic 53 kDa)
- Reads ground truth pixel intensities and expected molecule counts from `testdata/F_vs_amount.txt`
- Converts pixel intensities using `ProteinCalibration.pixel_intensities_to_molecules()`
- **Validation criteria**:
  - No negative molecule counts
  - Mean absolute error < 170 molecules

**Key features tested**:
- Loading calibration data with header rows
- Interpolation of calibration curves
- Pixel-to-arbitrary-unit conversion (factor: 1e7)
- Mass-to-molecule conversion using molecular weights
- Handling of both nuclear and cytosolic proteins

### 2. `fluorescence_extractor_test.py`

**Purpose**: Validate cell segmentation, tracking across timepoints, and fluorescence extraction.

**Test**: `test_track_cells()`
- Loads time-series images (intensity and phase contrast)
- Segments cells and tracks them across 144 timepoints (0-35.75 hours at 15-minute intervals)
- Validates cell positions and fluorescence extraction against ground truth

**Test data**: `testdata/F_vs_amount.txt`
- Contains ground truth for 80 cells per timepoint
- Columns:
  - `[0, 3]`: Nuclear and cytosolic fluorescence intensities
  - `[6]`: Cell indices
  - `[7]`: Cell ID flags
  - `[8:]`: Cell centroid positions (y, x)

**Validation criteria per timepoint**:

1. **Cell count**: Must detect exactly 80 cells
2. **Position accuracy** (tracking):
   - Uses Hungarian algorithm to match tracked cells to ground truth positions
   - At first timepoint (t=0): All cells must have position error < 7 pixels
   - Subsequent timepoints: Cell IDs must remain consistent with initial mapping
   - Position error threshold: 7 pixels
3. **Fluorescence extraction**:
   - Extracts nuclear and cytosolic intensities
   - Uses Hungarian algorithm to match fluorescence values
   - Validates both nuclear (`n_intensity`) and cytosolic (`c_intensity`)
   - Mean error threshold: < 5 intensity units
   - Standard deviation threshold: < 6 intensity units

**Key features tested**:
- Phase contrast image segmentation (`segment_cells`)
- Cell tracking across time (`track_cells_across_time`)
- Fluorescence extraction (`extract_nuclear_cytoplasmic`)
- Track consistency across timepoints
- Centroid calculation accuracy

### 3. `ode_inference_test.py`

**Purpose**: Validate the ODE model simulation and parameter inference.

**Test 1**: `test_repressillator_simulation_class()`
- Tests forward simulation of the Repressilator ODE model
- Loads true parameters from `testdata/protein_numbers/parameters/params_001.json`
- Loads expected simulation output from `testdata/protein_numbers/simulation_001.txt`
- Simulates using `RepressilatorModel.simulate()` with true parameters
- **Validation criteria**:
  - RMSE for nuclear protein (p1) < 26 molecules
  - RMSE for cytosolic protein (p2) < 26 molecules

**Test 2**: `test_infer_parameters()`
- Tests parameter inference (fitting) on synthetic data
- Extracts cell #1 data (80 timepoints) from ground truth
- Runs parameter inference using `infer_parameters()` with differential evolution
- Compares fitted parameters to true parameters
- Simulates with fitted parameters and checks fit quality
- **Validation criteria**:
  - RMSE for nuclear protein < 16 molecules
  - RMSE for cytosolic protein < 16 molecules
  - Prints standard errors for each fitted parameter

**Key features tested**:
- ODE system integration (`RepressilatorModel.simulate`)
- Parameter bounds and constraints
- Differential evolution optimization
- Goodness of fit (residual calculation)
- Model predictions with inferred parameters

### 4. `test_utils.py`

**Purpose**: Provides utility functions for test validation.

**Function**: `map_true_indices_to_tracks(true_positions, true_indices, tracks)`

Uses the Hungarian algorithm (linear sum assignment) to optimally match:
- True cell positions from ground truth
- Tracked cell positions from segmentation

**Returns**:
- `mapping`: dict[cell_id → true_index]
- `backmapping`: dict[true_index → cell_id]
- `cost_values`: list of assignment distances

**Use cases**:
- Validating tracking consistency across timepoints
- Identifying mismatched or lost tracks
- Computing position errors for quality control

## Ground Truth Data

### `testdata/F_vs_amount.txt`

**Format**: Text file with 11,520 rows (144 timepoints × 80 cells)

**Columns** (10 total):
- `[0]`: Nuclear fluorescence intensity (arbitrary units)
- `[1]`: Nuclear protein count (molecules, with +50 offset)
- `[2]`: (Possibly nuclear-related metadata)
- `[3]`: Cytosolic fluorescence intensity (arbitrary units)
- `[4]`: Cytosolic protein count (molecules, with +50 offset)
- `[5]`: (Possibly cytosolic-related metadata)
- `[6]`: Cell index (0-79 within each timepoint)
- `[7]`: Cell flag (used to identify specific cells in tests)
- `[8]`: Cell centroid Y position (row, pixels)
- `[9]`: Cell centroid X position (column, pixels)

**Note**: Protein counts have a +50 offset that is subtracted in tests:
```python
true_protein = data[:, [1, 4]] - 50
```

### `testdata/protein_numbers/`

**`simulation_001.txt`**: Synthetic ODE simulation
- Column 0: Time (seconds)
- Column 1: Nuclear protein p1 (molecules, +50 offset)
- Column 2: Cytosolic protein p2 (molecules, +50 offset)

**`parameters/params_001.json`**: True parameter values
```json
{
    "alpha": <transcription_rate>,
    "alpha0": <basal_transcription>,
    "beta": <translation_rate>,
    "hill": <cooperativity>,
    "mrna_half_life": <minutes>,
    "p_half_life": <minutes>
}
```

## Test Design Philosophy

### 1. **Ground Truth Validation**
All tests compare against pre-computed ground truth data rather than relying on relative comparisons. This ensures:
- Absolute accuracy standards
- Regression detection
- Reproducibility

### 2. **Component-Level Testing**
Each module is tested independently:
- Calibration conversion
- Cell segmentation and tracking
- ODE simulation
- Parameter inference

### 3. **Realistic Data**
Test data represents actual experimental conditions:
- 144 timepoints (35.75 hours)
- 80 cells per frame
- Realistic protein dynamics
- Real image segmentation challenges

### 4. **Quantitative Thresholds**
All validation uses explicit numerical thresholds:
- Position accuracy: < 7 pixels
- Fluorescence error: mean < 5, std < 6
- ODE simulation RMSE: < 26 molecules
- Parameter inference RMSE: < 16 molecules

## Interpreting Test Failures

### Calibration Test Failures
- **Negative molecules**: Check `PIXEL_TO_AU_FACTOR` or interpolation extrapolation
- **High mean error**: Verify calibration file loading, molecular weight values

### Tracking Test Failures
- **Wrong cell count**: Check `min_cell_area` parameter or segmentation thresholding
- **Position errors at t=0**: Segmentation issue, check phase image quality
- **Position errors at t>0**: Tracking algorithm failure, check overlap calculation
- **Inconsistent cell IDs**: Track assignment bug, verify overlap-based matching

### Fluorescence Extraction Failures
- **High mean error**: Check channel mapping (red vs green)
- **High std error**: Uneven fluorescence extraction or segmentation boundary issues

### ODE Test Failures
- **Simulation RMSE high**: ODE integration issue or parameter loading error
- **Inference RMSE high**: Optimization not converging, increase `maxiter` or try different method

## Adding New Tests

When adding tests, follow this structure:

```python
def test_new_feature():
    # 1. Load test data
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "...")
    data = np.loadtxt(testdata_path)

    # 2. Run the function being tested
    result = module.function(data)

    # 3. Load ground truth
    expected = data[:, some_columns]

    # 4. Assert quantitative criteria
    assert some_metric(result, expected) < threshold

    # 5. Optionally: detailed error reporting
    if failure:
        raise ValueError(f"Detailed error message with values")
```

## Coverage

Current test coverage:
- ✅ Calibration: Pixel to molecule conversion
- ✅ Image loading: Time-series organization
- ✅ Segmentation: Cell identification from phase images
- ✅ Tracking: Cell identity across timepoints
- ✅ Fluorescence extraction: Channel-specific intensities
- ✅ ODE simulation: Forward model
- ✅ Parameter inference: Optimization-based fitting
- ❌ Pipeline integration: End-to-end workflow (partial)
- ❌ Error handling: Edge cases and invalid inputs
- ❌ Visualization: Plot generation

## Performance Benchmarks

Typical test execution times (on standard hardware):
- `test_calibration`: ~0.1 seconds
- `test_track_cells`: ~30-60 seconds (processes 144 images)
- `test_repressillator_simulation_class`: ~0.5 seconds
- `test_infer_parameters`: ~60-120 seconds (optimization is slow)

**Note**: `test_track_cells` and `test_infer_parameters` are computationally intensive. Use `-v` flag to monitor progress.

## Troubleshooting

### "FileNotFoundError: testdata/..."
- Ensure you're running pytest from the repository root
- Check that testdata files are present and not gitignored

### "No module named 'repressilator_analysis'"
- Install package in editable mode: `pip install -e .`
- Verify virtual environment is activated

### Tests pass locally but fail in CI
- Check random seeds (if any stochastic operations)
- Verify library versions match requirements.txt
- Ensure image files are committed to repo

### Slow test execution
- Use `pytest -k "not infer"` to skip slow inference test
- Consider parallelization: `pytest -n auto` (requires pytest-xdist)
- Reduce MCMC iterations or optimization maxiter for faster debugging

## Future Improvements

Potential test enhancements:
1. Add tests for edge cases (empty images, single cells, etc.)
2. Add integration test for full `pipeline.full_analysis()`
3. Add tests for visualization functions
4. Mock external dependencies for faster unit tests
5. Add performance regression tests
6. Generate synthetic test images programmatically
7. Add tests for error handling and input validation
8. Test convergence diagnostics for parameter inference
9. Add tests for different optimization methods (L-BFGS-B vs differential evolution)

## Maintainer Notes

When updating the codebase:
1. Run full test suite before committing
2. Update ground truth data if algorithm changes fundamentally
3. Update tolerance thresholds if justified by methodology improvements
4. Document any changes to test data format
5. Keep test execution time reasonable (< 5 minutes total)

## Contact

For questions about tests or to report test failures, please open an issue on the repository.
