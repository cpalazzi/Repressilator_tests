import unittest
from unittest.mock import patch
import pytest
import numpy as np
import os
import repressilator_analysis as ra
import matplotlib.pyplot as plt
def test_calibration():
    calibrant_names=[ "Nuclear repressor 1 (66 kDa) calibration.txt","Cytosolic repressor (53 kDa) calibration.txt"]
    weights=[66, 53]
    calibrant_files=[os.path.join(os.path.dirname(__file__), os.pardir, "docs", x) for x in calibrant_names]
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata", "F_vs_amount.txt")
    all_true_data= np.loadtxt(testdata_path)
    for i in range(0, len(calibrant_files)):
        calibrator=ra.calibration.ProteinCalibration(calibrant_files[i], weights[i], header=1)
        raw_pixels=all_true_data[:,i*3]
        true_amounts=all_true_data[:,(i*3)+1]
        calculated_molecules=calibrator.pixel_intensities_to_molecules(raw_pixels)
        if any([x<0 for x in calculated_molecules]):
            raise ValueError("Negative molecules in pixel calculation")
        assert np.mean(np.abs(calculated_molecules-true_amounts))<170
test_calibration()