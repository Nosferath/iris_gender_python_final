from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment

from constants import FEMALES_LABEL, MALES_LABEL
from load_data_utils import apply_masks_to_data
from utils import Timer


def absolute_growth(mask_a, mask_b):
    """Calculates the absolute growth between two masks. When generating
    the pairs with this scoring_fn one should set maximize=False.
    """
    assert mask_a.size == mask_b.size, "Masks should have the same size"
    aub = np.any((mask_a, mask_b), axis=0)
    sum_aub = np.sum(aub)
    sum_a = np.sum(mask_a)
    sum_b = np.sum(mask_b)
    n = mask_a.size
    return_value = max((sum_aub - sum_a) / n,
                       (sum_aub - sum_b) / n)
    assert 0 <= return_value <= 1, \
        f"absolute growth should be between 0 and 1 ({return_value})"
    return return_value


def calculate_spp_matrix(female_masks, male_masks, scoring_fn=absolute_growth):
    n_fem = female_masks.shape[0]
    n_mal = male_masks.shape[0]

    spp_mat = np.zeros((n_fem, n_mal), dtype=float)
    for f_idx, m_idx in product(range(n_fem), range(n_mal)):
        cur_f = female_masks[f_idx, :]
        cur_m = male_masks[m_idx, :]
        spp_mat[f_idx, m_idx] = scoring_fn(cur_f, cur_m)

    return spp_mat


def generate_pairs(data_y, data_m, threshold=0.1, maximize=False,
                   spp_mat=None):
    """Generates pairs using the stored SPP Matrix. Only the images
    in img_names will be used. This is done by checking on the
    male_img_names and female_img_names attributes of the SPPMat.
    """
    female_masks = data_m[data_y == FEMALES_LABEL, :]
    female_idxs = np.where(data_y == FEMALES_LABEL)[0]
    male_masks = data_m[data_y == MALES_LABEL, :]
    male_idxs = np.where(data_y == MALES_LABEL)[0]
    if spp_mat is None:
        t = Timer('Spp mat calculation')
        t.start()
        spp_mat = calculate_spp_matrix(female_masks, male_masks)
        t.stop()
    spp_mat_compensated = spp_mat.copy()

    if spp_mat.shape[0] != spp_mat.shape[1]:
        raise ValueError('Data must be balanced before pairing')

    # Apply thresholds
    if maximize:
        bad_values = spp_mat[spp_mat < threshold]
        bad_values = -np.abs(np.divide(1e5, bad_values))
        spp_mat_compensated[spp_mat < threshold] = bad_values
    else:
        bad_values = spp_mat[spp_mat > threshold]
        bad_values = bad_values * 1e5
        spp_mat_compensated[spp_mat > threshold] = bad_values
    # Generate pairs
    females_pair_idx, males_pair_idx = linear_sum_assignment(
        spp_mat_compensated, maximize
    )
    final_values = spp_mat[females_pair_idx, males_pair_idx]
    # Get original indexes
    females_pair_idx = female_idxs[females_pair_idx]
    males_pair_idx = male_idxs[males_pair_idx]

    return np.array([females_pair_idx, males_pair_idx]), final_values


def apply_pairs(pairs, data_x, data_m):
    """Applies mask pairs to the data. Uses the apply_masks_to_data
    function, which scales non-masked data appropriately. The returned
    data is in the same scale (0-1 or 0-255) as the input data.

    Parameters
    ----------
    pairs : np.array
        Array with pairs, shape is [2, n_pairs]
    data_x : np.array
        Array with the iris data
    data_m : np.array
        Array with the mask data

    Returns
    -------
    np.array
        Iris data with paired masks properly applied
    """
    rescale = data_x.max() == 1
    n_pairs = pairs.shape[1]
    for i in range(n_pairs):
        cur_pair = pairs[:, i]
        cur_masks = data_m[cur_pair, :]
        cur_masks = np.any(cur_masks, axis=0) * 1
        data_m[cur_pair, :] = cur_masks
    data_x = apply_masks_to_data(data_x, data_m)
    if rescale:
        assert data_x.max() == 255, "apply masks does not scale to 0-255"
        data_x = data_x / 255

    return data_x


def remove_pairs(train_x, train_y, pairs, pair_values, threshold=0.1,
                 m_array=None, l_array=None):
    """Removes all pairs with scores strictly above the threshold."""
    good_pairs_idx_bool = pair_values <= threshold
    good_pairs_idx = pairs[:, good_pairs_idx_bool].flatten()
    good_pairs_idx.sort()
    train_x = train_x[good_pairs_idx, :]
    train_y = train_y[good_pairs_idx]
    to_return = [train_x, train_y]
    if m_array is not None:
        m_array = m_array[good_pairs_idx, :]
        to_return.append(m_array)
    if l_array is not None:
        l_array = l_array[good_pairs_idx]
        to_return.append(l_array)

    return to_return


def _test_remove_pairs():
    train_x = np.arange(10)
    train_x = np.tile(train_x[..., np.newaxis], (1, 20))
    train_y = np.repeat([0, 1], 5)
    pairs = np.array([range(5), range(9, 4, -1)])
    pair_values = np.arange(0.08, 0.13, 0.01)[::-1]
    train_x, train_y = remove_pairs(train_x, train_y, pairs, pair_values)
    assert (train_x[:, 0] == [2, 3, 4, 5, 6, 7]).all(), \
        "remove_pairs not working as intended"
