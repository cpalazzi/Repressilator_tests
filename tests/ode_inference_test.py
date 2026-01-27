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
    return np.sqrt(np.mean(np.square(np.subtract(y, y_data))))
def test_repressillator_simulation_class():
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
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    all_true_data= np.loadtxt(testdata_path)
    minutes=list(range(0, 2160, 15))
    true_array=np.zeros((len(minutes), 3))
    counter=0
    for j in range(0, len(all_true_data)):
        if all_true_data[j,7]==0:
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
