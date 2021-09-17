from pathlib import Path

import numpy as np
from scipy.io import loadmat

# DO NOT IMPORT LOCAL PACKAGES IN THIS MODULE


def prepare_pairs_indexes(pairs: np.ndarray) -> np.ndarray:
    """Takes the indexes from the pairs array, turns them to ints, and
    ensures they start from 0. It has no effect if the pairs array has
    been previously prepared.
    """
    pairs = pairs[:, :2].astype(int)
    if np.min(pairs) == 1:
        pairs = pairs - 1
    return pairs


def load_pairs_array(dataset_name: str, pair_method: str, partition: int):
    """Loads the pairs array from the .mat file. The array is fixed
    so it is zero-indexed for use with Python and numpy."""
    pairs_path = Path.cwd() / 'pairs' / pair_method / dataset_name
    pairs_mat = loadmat(str(pairs_path / (str(partition) + '.mat')))
    pairs = pairs_mat['pairs']
    pairs[:, [0, 1]] = pairs[:, [0, 1]] - 1
    return pairs
