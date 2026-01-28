import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

@pytest.fixture(scope="module")
def test_images(request):
    """Load test images once and share across all tests in this module."""
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")

    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    all_true_data= np.loadtxt(testdata_path)
   
    return {
        'timepoints': timepoints,
        'intensity_images': intensity_images,
        'phase_images': phase_images,
        "data":all_true_data
    }

class TestFluorescenceIntegration:
    """Integration tests for fluorescence extraction pipeline."""
    def test_track_cells(self, test_images):
        tracks=ra.fluorescence_extraction.track_cells_across_time(test_images["phase_images"], 5)
        if len(tracks)==2:
            labelled_images=tracks[1]
            new_tracks=[[] for x in range(0, len(test_images["phase_images"]))]
            for i in range(0, len(test_images["phase_images"])):
                for key in tracks[0].keys():
                    for elem in tracks[0][key]:
                        if elem[0]==i:
                            new_tracks[i].append({"cell_id":int(key), "centre":list(elem[2])})
            tracks=new_tracks

        actual_data=test_images["data"]
        intensity_images = test_images['intensity_images']

        for i in range(0, len(tracks)):
            positions=actual_data[i*80:(i+1)*80,8:]
            indices=list(actual_data[i*80:(i+1)*80,6])
            if len(tracks[i])!=80:
                raise ValueError(f"At timepoint {i}, the number of cells in the segmented image is {len(tracks[i])}, not 80")

            # Test cell tracking (positions)
            cost_matrix=np.zeros((len(tracks[i]), len(tracks[i])))
            for r in range(0, positions.shape[0]):
                for q in range(0, len(tracks[i])):
                    cost_matrix[r, q]=np.linalg.norm(positions[r,:]-tracks[i][q]["centre"])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            actual_cost_values=[cost_matrix[x,y] for x,y in zip(row_ind, col_ind)]
            errors=[x>7 for x in actual_cost_values]
            num_errors=len([x for x in errors if x is True])
            if i==0:
                if any(errors):
                    raise ValueError(f"At the first timepoint, {num_errors} cell centres have been assigned incorrectly")
                mapping={tracks[i][y]["cell_id"]:indices[x] for x,y in zip(row_ind, col_ind)}
                backmapping={indices[x]:tracks[i][y]["cell_id"] for x,y in zip(row_ind, col_ind)}
            else:
                for j in range(0, len(col_ind)):
                    cidx=col_ind[j]
                    #improper assignment.
                    if mapping[tracks[i][cidx]["cell_id"]]!=indices[row_ind[j]]:
                        raise ValueError(f"At time {i}, cell_id {backmapping[indices[row_ind[j]]]} (true position {positions[row_ind[j]]}) has been incorectly assigned to cell_id {tracks[i][cidx]["cell_id"]}")
                    distance=actual_cost_values[j]
                    if distance>7:
                        raise ValueError(f"At time {i}, cell_id {backmapping[indices[row_ind[j]]]} (true position {positions[row_ind[j]]}) has been given the position {tracks[i][col_ind[j]]["centre"]}, (error:{actual_cost_values[j]:.2f} pixels)")

            # Test fluorescence extraction
            results = ra.fluorescence_extraction.extract_nuclear_cytoplasmic(
                intensity_images[i],
                tracks[i],
            )

            # Determine which keys are used for fluorescence data
            if "cytoplasmic" in results[0]:
                keys = ["nuclear", "cytoplasmic"]
            elif "n_intensity" in results[0]:
                keys = ["n_intensity", "c_intensity"]
            else:
                raise KeyError(f"Expected fluorescence keys not found in segmented data. Available keys: {results[0].keys()}")

            # Extract test values (nucleus, cell) for each segment
            test_vals = np.column_stack((
                np.array([s[keys[0]] for s in results]),
                np.array([s[keys[1]] for s in results])
            ))
            test_fs = actual_data[:, [0, 3]]

            # Check fluorescence distances are below thresholds
            for j in range(0, 2):
                fluor_cost_matrix = np.abs(test_vals[:, j][:, np.newaxis] - test_fs[i*80:(i+1)*80, j][np.newaxis, :])
                fluor_row_ind, fluor_col_ind = linear_sum_assignment(fluor_cost_matrix)
                best_distances = [fluor_cost_matrix[x] for x in zip(fluor_row_ind, fluor_col_ind)]

                if np.mean(best_distances) > 5 or np.std(best_distances) > 6:
                    raise ValueError(f"At timepoint {i}, fluorescence value extraction in {keys[j]} over distance threshold to true value (mean difference {np.mean(best_distances)}, s.d. {np.std(best_distances)})")

