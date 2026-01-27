"""
Calibration utilities for converting fluorescence intensities to protein quantities.

This module provides a ProteinCalibration class to load calibration data and convert pixel
intensities to protein mass (nanograms) or molecule counts.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Union


# Conversion factor between pixel intensities and arbitrary units in calibration
PIXEL_TO_AU_FACTOR = 1e7

# Avogadro's number
AVOGADRO = 6.02214076e23


class ProteinCalibration:
    """
    Calibration curve for converting fluorescence to protein quantity.

    Loads calibration data from a file and provides conversion from pixel
    intensities to molecule counts.
    """

    def __init__(self, filepath: str, molecular_weight_kda: float, header: int = 1):
        """
        Initialize calibration curve from a calibration file.

        Args:
            filepath: Path to calibration file
            molecular_weight_kda: Molecular weight of the protein in kDa
            header: Number of header rows to skip (default: 1)
        """
        self.filepath = filepath
        self.molecular_weight_kda = molecular_weight_kda
        self.header = header

        # Load calibration data
        self.mass_ng, self.fluorescence_au = self._load_calibration_file()

        # Create interpolation function (fluorescence -> mass)
        self.interp_func = interp1d(
            self.fluorescence_au,
            self.mass_ng,
            kind='linear',
            fill_value='extrapolate'
        )

    def _load_calibration_file(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load calibration data from the file.

        Expected format:
            Mass/ng repeat 1/A.U. repeat 2/A.U. repeat 3/A.U.
            2.500e+01 5.969e+02 6.228e+02 8.278e+02
            ...

        Returns:
            Tuple of (mass_ng, mean_fluorescence_au)
        """
        data = np.loadtxt(self.filepath, skiprows=self.header)

        mass_ng = data[:, 0]
        repeats = data[:, 1:]

        # Average across repeats
        mean_fluorescence_au = np.mean(repeats, axis=1)

        return mass_ng, mean_fluorescence_au

    def pixel_intensities_to_molecules(
        self,
        pixel_intensities: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Convert pixel intensities to number of molecules.

        Args:
            pixel_intensities: Pixel intensity value(s) from image

        Returns:
            Number of molecules (same shape as input)
        """
        # Convert to numpy array for vectorized operations
        pixel_intensities = np.asarray(pixel_intensities)

        # Convert pixel intensities to fluorescence in arbitrary units
        fluorescence_au = pixel_intensities * PIXEL_TO_AU_FACTOR

        # Convert fluorescence to mass in nanograms
        mass_ng = self.interp_func(fluorescence_au)

        # Convert ng to grams
        mass_g = mass_ng * 1e-9

        # Convert kDa to g/mol
        molecular_weight_g_per_mol = self.molecular_weight_kda * 1000

        # Calculate moles
        moles = mass_g / molecular_weight_g_per_mol

        # Convert to molecules
        molecules = moles * AVOGADRO

        return molecules
