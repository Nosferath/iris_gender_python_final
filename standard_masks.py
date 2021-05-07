import numpy as np


def generate_standard_masks(mask_array: np.ndarray, pairs: np.ndarray):
    """From the pairs and mask array, generate a new mask array
    where masks are standardized (combining each pair). Pairs
    have to be zero-indexed (use load_pairs_array)."""
    std_masks = mask_array.copy()
    n_pairs = pairs.shape[0]

    for i in range(n_pairs):
        cur_pairs = pairs[i, :2].astype(int)
        cur_masks = mask_array[cur_pairs, :]
        std_mask = np.any(cur_masks, axis=0)
        std_masks[cur_pairs, :] = std_mask
    return std_masks


def apply_std_mask(x_array: np.ndarray, std_masks: np.ndarray, mask_value):
    masked_x_array = x_array.copy()
    masked_x_array[std_masks == 1] = mask_value
    return masked_x_array
