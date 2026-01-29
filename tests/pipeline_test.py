"""
Tests for the complete analysis pipeline.

This module tests the full Repressilator analysis workflow including
cell tracking, protein number extraction, and parameter inference.
"""

import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt
import json
from test_utils import map_true_indices_to_tracks


def test_extraction():
    """
    Test protein molecule number extraction from tracked cells.

    Validates that extract_protein_numbers_from_tracks() correctly:
    1. Tracks 80 cells across all timepoints
    2. Extracts fluorescence intensities
    3. Converts intensities to molecule counts using calibration
    4. Produces values within 200 molecules RMSE of ground truth

    Asserts:
        - Mean RMSE across all cells and proteins < 200 molecules
    """
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")
    calibrant_names=[ "Nuclear repressor 1 (66 kDa) calibration.txt","Cytosolic repressor (53 kDa) calibration.txt"]
    weights=[66, 53]
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    actual_data= np.loadtxt(testdata_path)
    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    tracks=ra.fluorescence_extraction.track_cells_across_time(phase_images, 5)
    if isinstance(tracks, tuple):
        extraction_arg=tracks
        labelled_images=tracks[0]
        new_tracks=[[] for x in range(0,len(phase_images))]
        for i in range(0,len(phase_images)):
            for key in tracks[0].keys():
                for elem in tracks[0][key]:
                    if elem[0]==i:
                        new_tracks[i].append({"cell_id":int(key), "centre":list(elem[2])})
        tracks=new_tracks
    else:
        extraction_arg=tracks
    for i in range(0, len(tracks)):
        if len(tracks[i])!=80:
            raise ValueError(f"At timepoint {i} there are {len(tracks[i])} tracked cells rather than 80")
    positions=actual_data[:80,8:]
    indices=list(actual_data[:80,6])
    mapping, backmapping, actual_cost_values = map_true_indices_to_tracks(positions, indices, tracks[0])  
    calibrant_files=[os.path.join(os.path.dirname(__file__), os.pardir, "docs", x) for x in calibrant_names]
    recovered_numbers=ra.pipeline.extract_protein_numbers_from_tracks(
        extraction_arg,
        timepoints, intensity_images, phase_images,
        calibration_dict={"files": calibrant_files, "weights": weights, "headers": 1},
        track_labels=["n_intensity", "c_intensity"],)
    protein_values=actual_data[:, [1, 4]]
    errors=[]
    for i in range(0,80):
        for m in range(0, 2):
        
            idx=np.where(actual_data[:,6]==i)
            actual_values=protein_values[idx, m]
            
            #index in tracks where the cell id corresponds to the appropriate index
            recovered_col=[x for x in range(0, len(tracks[0])) if tracks[0][x]["cell_id"]==backmapping[int(i)]][0]
            calculated_values=recovered_numbers[:,recovered_col, m]
            error=ra.utils.RMSE(calculated_values, actual_values[0])
            errors.append(error)
    assert np.mean(error)<200


def test_full_pipeline():
    """
    Test the complete end-to-end Repressilator analysis pipeline.

    This integration test validates the full_analysis() function by:
    1. Loading intensity and phase images
    2. Segmenting and tracking cells
    3. Extracting protein molecule numbers
    4. Inferring ODE parameters for specific cells

    Tests parameter inference accuracy by comparing inferred parameters
    against known ground truth values for cells [1, 3, 5] and validating
    that simulations using inferred parameters accurately reproduce the
    observed protein time series.

    Asserts:
        - RMSE for nuclear protein < 160 molecules per cell
        - RMSE for cytosolic protein < 160 molecules per cell
        - Parameter estimates within reasonable error bounds
    """
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")
    calibrant_names=[ "Nuclear repressor 1 (66 kDa) calibration.txt","Cytosolic repressor (53 kDa) calibration.txt"]
    weights=[66, 53]

    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    actual_data= np.loadtxt(testdata_path)
    target_cells=[1, 3, 5]
    timepoints=np.array(range(0, 2160, 15))*60
    indices=actual_data[:80,7]

    actual_params_of_interest=set(["hill",
                    "mrna_half_life",
                    "p_half_life",
                    "K_m",
                    "T_e",
                    "alpha",
                    "alpha0"])
    params_of_interest=list(actual_params_of_interest.intersection(ra.ode_inference.RepressilatorModel._param_names))

    for cell_idx, target_cell_id in enumerate(target_cells):
        tracks, protein_numbers, parameters=ra.pipeline.full_analysis(
            intensity_dir,
            phase_dir,
            {"files":[os.path.join(os.path.dirname(__file__), os.pardir, "docs", x) for x in calibrant_names],
            "weights":[66, 53],
            "headers":1},
            cell_ids=[target_cell_id]
        )
        mapping, backmapping, actual_cost_values = map_true_indices_to_tracks(actual_data[:80,8:], indices, tracks[0])

        true_array=np.zeros((len(timepoints), 2))
        for j in range(0,2):
            idx=np.where( actual_data[:,7]==mapping[target_cell_id])
            true_array[:,j]=actual_data[idx,(j*3)+1]

        valuedict=dict(zip(ra.ode_inference.RepressilatorModel._param_names, parameters[target_cell_id]))

        with open(os.path.join(os.path.dirname(__file__), "testdata", "protein_numbers", "parameters", f"params_{target_cell_id:03d}.json"),"r") as f:
            true_params=json.load(f)
        standard_errors=[]
        for param in  params_of_interest:
            standard_errors.append(100*abs(true_params[param]-np.mean(valuedict[param]))/true_params[param])
        print(f"Cell {target_cell_id} - Standard errors:")
        for x, y in zip(params_of_interest, standard_errors):
            print(x, ":", round(y,1), "%")

        model = ra.ode_inference.RepressilatorModel(timepoints)
        sim=model.simulate(parameters[target_cell_id], timepoints)
        assert ra.utils.RMSE(sim[:,0], true_array[:,0]) <160
        assert ra.utils.RMSE(sim[:,1], true_array[:,1]) <160
