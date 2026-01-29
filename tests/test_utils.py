"""Utility functions for testing."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def map_true_indices_to_tracks(true_positions, true_indices, tracks):
    """Map true cell indices to track cell IDs based on position matching.

    Uses the Hungarian algorithm (linear_sum_assignment) to find the optimal
    assignment between true cell positions and tracked cell positions based
    on Euclidean distance.

    Parameters
    ----------
    true_positions : np.ndarray
        Array of shape (n_cells, 2) containing true cell center positions.
        Typically extracted from F_vs_amount data columns [8:].
    true_indices : list or np.ndarray
        List of true cell indices corresponding to each position.
        Typically extracted from F_vs_amount data column 6 or 7.
    tracks : list of dict
        List of tracked cells, where each dict has keys "cell_id" and "centre".

    Returns
    -------
    mapping : dict
        Dictionary mapping cell_id (from tracks) -> true_index (from F_vs_amount).
    backmapping : dict
        Dictionary mapping true_index (from F_vs_amount) -> cell_id (from tracks).
    cost_values : list
        List of distance values for each assignment, useful for validation.

    Examples
    --------
    """
    # Build cost matrix based on Euclidean distance
    cost_matrix = np.zeros((len(true_positions), len(tracks)))
    for r in range(true_positions.shape[0]):
        for q in range(len(tracks)):
            cost_matrix[r, q] = np.linalg.norm(true_positions[r, :] - tracks[q]["centre"])

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get actual cost values for validation
    actual_cost_values = [cost_matrix[x, y] for x, y in zip(row_ind, col_ind)]

    # Create bidirectional mappings
    mapping = {tracks[y]["cell_id"]: true_indices[x] for x, y in zip(row_ind, col_ind)}
    backmapping = {true_indices[x]: tracks[y]["cell_id"] for x, y in zip(row_ind, col_ind)}

    return mapping, backmapping, actual_cost_values
