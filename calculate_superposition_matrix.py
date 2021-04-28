import numpy as np


def calculate_mask_iou(mask_a, mask_b):
    assert mask_a.shape == mask_b.shape, 'Masks must be the same shape'
    mask_a = mask_a != 0
    mask_b = mask_b != 0
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    union = np.sum(np.logical_or(mask_a, mask_b))
    return intersection / union


def calculate_superposition_matrix(mask_array):
    n_masks = mask_array.shape[0]
    spp_mat = np.zeros((n_masks, n_masks))
    for i in range(n_masks):
        mask_a = mask_array[i, :]
        for j in range(i, n_masks):
            mask_b = mask_array[j, :]
            spp_mat[i, j] = calculate_mask_iou(mask_a, mask_b)
            spp_mat[j, i] = spp_mat[i, j]
    return spp_mat
