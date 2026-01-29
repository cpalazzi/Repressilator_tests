"""
ODE model and parameter inference for the Repressilator system.

This module implements the Repressilator ODE model and uses PINTS
for Bayesian parameter inference.
"""

import numpy as np
import pints
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from typing import List, Tuple, Dict, Optional


class RepressilatorModel(pints.ForwardModel):
    """
    ODE model for the Repressilator genetic circuit.

    The Repressilator consists of three repressors (LacI, TetR, CI) that
    form a cyclic negative feedback loop. Each repressor inhibits the
    transcription of the next repressor in the cycle.

    State variables:
    - m1, m2, m3: mRNA concentrations for repressors 1, 2, 3
    - p1, p2, p3: Protein concentrations for repressors 1, 2, 3

    Parameters:
    - alpha: Transcription rate
    - alpha0: Basal transcription rate
    - beta: Translation rate
    - hill: Hill coefficient (cooperativity)
    - mrna_half_life: mRNA half-life (minutes)
    - p_half_life: Protein half-life (minutes)
    """
    _param_names=["alpha", "alpha0", "beta", "hill", "mrna_half_life", "p_half_life"]
    def __init__(self, times: np.ndarray):
        """
        Initialize the Repressilator model.

        Args:
            times: Array of time points for simulation
        """
        self.times = times

    def n_parameters(self) -> int:
        """Return the number of model parameters."""
        return 6  # [alpha, alpha0, beta, hill, mrna_half_life, p_half_life]

    def n_outputs(self) -> int:
        """Return the number of observable outputs."""
        # We observe 2 proteins (nuclear and cytoplasmic)
        return 2

    def simulate(self, parameters: List[float], times: np.ndarray) -> np.ndarray:
        """
        Simulate the Repressilator ODE system.

        Args:
            parameters: Model parameters [alpha, alpha0, beta, hill, mrna_half_life, p_half_life]
            times: Time points for simulation

        Returns:
            Array of shape (n_times, n_outputs) with protein concentrations
        """
        alpha, alpha0, beta, hill, mrna_half_life, p_half_life = parameters

        # Convert half-lives to degradation rates: degradation_rate = ln(2) / half_life
        gamma_m = np.log(2) / mrna_half_life
        gamma_p = np.log(2) / p_half_life

        # Initial conditions (start at equilibrium estimate)
        y0 = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]  # [m1, m2, m3, p1, p2, p3]

        def repressilator_odes(y, t):
            """ODE system for the Repressilator."""
            m1, m2, m3, p1, p2, p3 = y

            # Hill function for repression
            def hill_repression(repressor_conc):
                return alpha / (1 + (repressor_conc ** hill)) + alpha0

            # mRNA dynamics
            dm1_dt = hill_repression(p3) - gamma_m * m1
            dm2_dt = hill_repression(p1) - gamma_m * m2
            dm3_dt = hill_repression(p2) - gamma_m * m3

            # Protein dynamics
            dp1_dt = beta * m1 - gamma_p * p1
            dp2_dt = beta * m2 - gamma_p * p2
            dp3_dt = beta * m3 - gamma_p * p3

            return [dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt]

        # Solve ODE system
        solution = odeint(repressilator_odes, y0, times)

        # Extract observable proteins (p1 and p2 as nuclear and cytoplasmic)
        # Note: p3 is the unobserved protein without fluorescence
        output = solution[:, [3, 4]]  # [p1, p2]

        return output


def infer_parameters(
    times: np.ndarray,
    observations: np.ndarray,
    method: str = 'differential_evolution',
) -> np.ndarray:
    """
    Infer Repressilator parameters using optimization.

    Args:
        times: Time points (in minutes)
        observations: Observed protein concentrations of shape (n_times, 2)
        method: Optimization method ('differential_evolution' or 'least_squares')

    Returns:
        Best-fit parameters [alpha, alpha0, beta, hill, mrna_half_life, p_half_life]
    """
    # Create model
    model = RepressilatorModel(times)

    # Define parameter bounds
    bounds = [
        (0, 1000),    # alpha
        (0, 10),      # alpha0
        (0, 100),     # beta
        (1, 5),       # hill
        (0.693, 69.3),  # mrna_half_life
        (6.93, 693),    # p_half_life
    ]

    # Define cost function (sum of squared residuals)
    def cost_function(parameters):
        try:
            predictions = model.simulate(parameters, times)
            residuals = observations - predictions
            return np.sum(residuals ** 2)
        except Exception:
            return np.inf

    print(f"Running optimization using {method}...")

    if method == 'differential_evolution':
        # Use differential evolution (global optimization)
        result = differential_evolution(
            cost_function,
            bounds,
            maxiter=1000,
            popsize=15,
            tol=1e-7,
            seed=42,
            disp=True
        )
        best_params = result.x

    else:  # least_squares or other local methods
        # Initial guess
        x0 = [100, 1, 10, 2, 6.93, 69.3]

        result = minimize(
            cost_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 1000}
        )
        best_params = result.x

    parameter_names = ['alpha', 'alpha0', 'beta', 'hill', 'mrna_half_life', 'p_half_life']

    print("\nBest-fit parameters:")
    for name, value in zip(parameter_names, best_params):
        print(f"  {name}: {value:.4f}")

    print(f"\nFinal cost (SSR): {cost_function(best_params):.4e}")

    return best_params


def run_inference_for_cell(
    times: np.ndarray,
    cell_data: Dict[str, List[float]],
    method: str = 'differential_evolution',
) -> Dict[str, any]:
    """
    Run ODE parameter inference for a single cell's time-series data.

    This function prepares observation data from a single cell and runs
    parameter optimization to find the best-fit Repressilator parameters.

    Args:
        times: Array of time points in minutes
        cell_data: Dictionary with keys 'nuclear' and 'cytoplasmic' containing
                   lists/arrays of protein concentration measurements over time
        method: Optimization method ('differential_evolution' or 'least_squares')

    Returns:
        Dictionary with keys:
        - 'times': Input time array
        - 'observations': 2D array of shape (n_times, 2) with nuclear and cytoplasmic data
        - 'best_fit_parameters': Array of 6 fitted parameters
        - 'parameter_names': List of parameter names

    Raises:
        ValueError: If cell_data is missing 'nuclear' or 'cytoplasmic' measurements
    """
    # Prepare observations
    nuclear = np.array(cell_data.get('nuclear', []))
    cytoplasmic = np.array(cell_data.get('cytoplasmic', []))

    if len(nuclear) == 0 or len(cytoplasmic) == 0:
        raise ValueError("Cell data must contain 'nuclear' and 'cytoplasmic' measurements")

    observations = np.column_stack([nuclear, cytoplasmic])

    # Run inference
    best_params = infer_parameters(times, observations, method)

    results = {
        'times': times,
        'observations': observations,
        'best_fit_parameters': best_params,
        'parameter_names': ['alpha', 'alpha0', 'beta', 'hill', 'mrna_half_life', 'p_half_life'],
    }

    return results
