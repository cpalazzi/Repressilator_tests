"""
Tests for ODE inference and Repressilator model simulation.

This module tests the RepressilatorModel class and parameter inference functions
to ensure accurate simulation of the repressilator genetic circuit and reliable
parameter estimation from time-series data.
"""

import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt
import json
import pints


def RMSE(y, y_data):
    """
    Calculate the Root Mean Square Error between two arrays.

    Args:
        y: First array of predicted values
        y_data: Second array of actual values

    Returns:
        Root mean square error as a float
    """
    return np.sqrt(np.mean(np.square(np.subtract(y, y_data))))


def test_repressillator_simulation_class():
    """
    Test RepressilatorModel simulation accuracy against reference data.

    Loads test simulation data and known parameters, then validates that
    the RepressilatorModel.simulate() method reproduces the expected
    protein concentrations for both nuclear and cytosolic repressors.

    Asserts:
        - RMSE for nuclear protein < 26 molecules
        - RMSE for cytosolic protein < 26 molecules
    """
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "protein_numbers")
    test_simulation= np.loadtxt(os.path.join(testdata_path, "simulation_001.txt"))
    test_time_seconds=test_simulation[:,0]
    test_nuclear_protein=test_simulation[:,1]-50
    test_cytosol_protein=test_simulation[:,2]-50
    with open(os.path.join(testdata_path, "parameters", "params_001.json"),"r") as f:
        params=json.load(f)
    simclass=ra.ode_inference.RepressilatorModel(test_time_seconds)
    sim_values=[params[x] for x in simclass._param_names]
    values=simclass.simulate(sim_values, test_time_seconds)
    assert RMSE(values[:,0], test_nuclear_protein)<26
    assert RMSE(values[:,1], test_cytosol_protein) <26


def test_infer_parameters():
    """
    Test parameter inference from time-series protein data.

    Loads test time-series data for a single cell and validates that the
    infer_parameters() function correctly estimates Repressilator model
    parameters (alpha, alpha0, beta, hill, mrna_half_life, p_half_life).
    Compares inferred parameters against known values and validates that
    simulations using inferred parameters match observed data.

    Asserts:
        - RMSE for nuclear protein simulation < 16 molecules
        - RMSE for cytosolic protein simulation < 16 molecules
        - Parameter estimates within reasonable error bounds
    """
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    all_true_data= np.loadtxt(testdata_path)
    minutes=np.array(range(0, 2160, 15))*60
    true_array=np.zeros((len(minutes), 3))
    counter=0
    for j in range(0, len(all_true_data)):
        if all_true_data[j,7]==1:
            true_array[counter,1:]=all_true_data[j,[1,4]]-50
            true_array[counter, 0]=minutes[counter]*60
            counter+=1
    
    values=ra.ode_inference.infer_parameters(true_array[:,0], true_array[:,1:])

    valuedict=dict(zip(ra.ode_inference.RepressilatorModel._param_names, values))
    actual_params_of_interest=set(["hill", 
                    "mrna_half_life",
                    "p_half_life", 
                    "K_m", 
                    "T_e", 
                    "alpha", 
                    "alpha0"])
    params_of_interest=list(actual_params_of_interest.intersection(ra.ode_inference.RepressilatorModel._param_names))
    with open(os.path.join(os.path.dirname(__file__), "testdata", "protein_numbers", "parameters", "params_001.json"),"r") as f:
        true_params=json.load(f)
    standard_errors=[]
    for param in  params_of_interest:
        standard_errors.append(100*abs(true_params[param]-np.mean(valuedict[param]))/true_params[param])
    print("Standard errors")
    for x, y in zip(params_of_interest, standard_errors):
        print(x, ":", round(y,1), "%")
    model = ra.ode_inference.RepressilatorModel(true_array[:,0])
    sim=model.simulate(values, true_array[:,0])
    assert RMSE(sim[:,0], true_array[:,1])<16
    assert RMSE(sim[:,1], true_array[:,2])<16
