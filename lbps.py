from typing import Tuple

import numpy as np
from skimage.feature import local_binary_pattern


def lbp1d(x_row: np.ndarray, n_points: int, radius: float, method: str,
          shape: Tuple[int]):
    """Calculates the LBP of the x_row, expecting it to be 1D, reshaping
    it to shape before calculating, and then flattening back. For using
    with np.apply_along_axis.
    """
    x_row = x_row.reshape(shape)
    out_lbp = local_binary_pattern(x_row, n_points, radius, method)
    return out_lbp.flatten()


def calculate_dataset_lbp(x_arr: np.ndarray, n_points: int, radius: float,
                          method: str):
    """Turns the entire dataset into LBPs of each row."""
    shapes = {4800: (20, 240), 9600: (40, 240), 38400: (80, 480)}
    shape = shapes[x_arr.shape[1]]
    out_arr = np.apply_along_axis(lbp1d, 1, x_arr, n_points, radius, method,
                                  shape)
    return out_arr


def calculate_hist_lbp(x_row: np.ndarray, max_value: int) -> np.ndarray:
    """Calculates the histogram for the array-row. Meant to be used with
    np.apply_along_axis.
    """
    return np.histogram(x_row, bins=np.arange(max_value+1), density=True)[0]


def convert_dataset_lbp_to_hist(x_arr, m_arr: np.ndarray = None,
                                remove_masks: bool = False):
    """Converts the lbp dataset (x_arr) to histograms. If a mask array
    is passed as m_arr, masks will be set to their own bin. Additionally
    if remove masks is true, this last bin will not be included."""
    max_value = x_arr.max()
    if m_arr is not None:
        # Set masked values to an additional bin
        x_arr = x_arr.copy()
        x_arr[m_arr == 1] = max_value + 1
        max_value += 1
    # Calculate histograms
    out_arr = np.apply_along_axis(calculate_hist_lbp, 1, x_arr, max_value)
    if m_arr is not None and remove_masks:
        # Remove masks (optional)
        out_arr = out_arr[:, :-1]
        # Re-scale the rest (density)
        sums = out_arr.sum(axis=1, keepdims=True)
        out_arr = out_arr / sums

    return out_arr
