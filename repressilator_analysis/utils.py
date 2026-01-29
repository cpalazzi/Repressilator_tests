"""
Utility functions for image processing and analysis.

This module provides helper functions for mask manipulation, centroid calculation,
position checking, and error metrics.
"""

import numpy as np


def map_mask_to_image(mask, image, value):
    """
    Apply a mask to an image by setting masked pixels to a specific value.

    Args:
        mask: Array of shape (n_pixels, 2) containing row and column indices
        image: Image array to modify
        value: Value to assign to masked pixels

    Returns:
        Copy of the image with masked pixels set to the specified value
    """
    rows = mask[:, 0]
    cols = mask[:, 1]
    copy_img = image.copy()
    copy_img[rows, cols] = value
    return copy_img


def get_centroid(mask):
    """
    Calculate the centroid (center of mass) of a mask.

    Args:
        mask: Array of shape (n_pixels, 2) containing row and column indices

    Returns:
        List [centroid_row, centroid_col] with centroid coordinates
    """
    centroid_row = np.mean(mask[:, 0])
    centroid_col = np.mean(mask[:, 1])
    return [centroid_row, centroid_col]


def check_position_dupes(centroid, centroid_list, threshold=3):
    """
    Check if a centroid is too close to any existing centroids.

    Used to detect duplicate cells or overlapping segmentation results.

    Args:
        centroid: Array-like of length 2 containing [row, col] coordinates
        centroid_list: List of existing centroids to compare against
        threshold: Maximum distance (in pixels) to consider positions as duplicates (default: 3)

    Returns:
        True if any existing centroid is within threshold distance, False otherwise
    """
    distances = [np.linalg.norm(centroid - x) for x in centroid_list]
    return any([x < threshold for x in distances])


def RMSE(x, y):
    """
    Calculate the Root Mean Square Error between two arrays.

    Args:
        x: First array of values
        y: Second array of values (must have same shape as x)

    Returns:
        Root mean square error as a float
    """
    return np.sqrt(np.mean(np.square(np.subtract(x, y))))