import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
def test_extraction():
    intensity_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "intensity")
    phase_dir = os.path.join(os.path.dirname(__file__), os.pardir, "images", "phase")
    calibrant_names=[ "Nuclear repressor 1 (66 kDa) calibration.txt","Cytosolic repressor (53 kDa) calibration.txt"]
    weights=[66, 53]
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    actual_data= np.loadtxt(testdata_path)
    timepoints, intensity_images, phase_images = ra.image_loader.load_timeseries(
        intensity_dir, phase_dir
    )
    tracks=ra.fluorescence_extraction.track_cells_across_time(phase_images[:2], 5)
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
    cost_matrix=np.zeros((len(tracks[0]), len(tracks[0])))
    for r in range(0, positions.shape[0]):
        for q in range(0, len(tracks[0])):
            cost_matrix[r, q]=np.linalg.norm(positions[r,:]-tracks[0][q]["centre"])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    actual_cost_values=[cost_matrix[x,y] for x,y in zip(row_ind, col_ind)]
    mapping={tracks[0][y]["cell_id"]:indices[x] for x,y in zip(row_ind, col_ind)}   
    backmapping={indices[x]:tracks[0][y]["cell_id"] for x,y in zip(row_ind, col_ind)}  
    print(backmapping)
    calibrant_files=[os.path.join(os.path.dirname(__file__), os.pardir, "docs", x) for x in calibrant_names]
    recovered_numbers=ra.pipeline.extract_protein_numbers_from_tracks(
        extraction_arg,
        intensity_dir, phase_dir, calibration_files=calibrant_files,weights=weights,
        track_labels=["n_intensity", "c_intensity"],calibration_headers=1,
        output_dir=None)
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


    
test_extraction()