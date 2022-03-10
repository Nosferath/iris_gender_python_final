from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from constants import FEMALES_LABEL, MALES_LABEL, SPP_FOLDER
from load_data_utils import apply_masks_to_data


def absolute_growth(mask_a, mask_b):
    """Calculates the absolute growth between two masks. When generating
    the pairs with this scoring_fn one should set maximize=False.
    """
    assert mask_a.size == mask_b.size, "Masks should have the same size"
    aub = np.any((mask_a, mask_b), axis=0)
    sum_aub = np.sum(aub)
    sum_a = np.sum(mask_a)
    sum_b = np.sum(mask_b)
    n = aub.size
    return max((sum_aub - sum_a) / n,
               (sum_aub - sum_b) / n)


class SPPMat:
    """This class keeps track of the SPP matrix as well as the
    identities of the compared iris, in order to retrieve them after
    generating the pairs.
    """
    def __init__(self, dataset_name):
        # Axis 0 in spp_mat is females, axis 1 is males
        self.spp_mat_path = Path(SPP_FOLDER) / f'{dataset_name}.npz'
        if self.spp_mat_path.exists():
            npz = np.load(self.spp_mat_path)
            self.spp_mat = npz['spp_mat']
            self.female_img_names = npz['female_img_names']
            self.male_img_names = npz['male_img_names']
        else:
            self.spp_mat: np.array = None
            self.female_img_names = None
            self.male_img_names = None
    
    def _save_data(self):
        """Saves the SPP Mat data into .npz"""
        self.spp_mat_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            self.spp_mat_path,
            spp_mat=self.spp_mat,
            female_img_names=self.female_img_names,
            male_img_names=self.male_img_names
        )

    def calculate_spp_matrix(
        self,
        labels_array,
        masks_array,
        img_names,
        scoring_fn=absolute_growth,
    ):
        """Calculates the SPP matrix for the given labels and masks
        arrays. If this has already been done, it just returns the
        stored spp_mat.
        """
        assert labels_array.size == masks_array.shape[0], \
            "There must be a single label per mask."
        assert labels_array.size == img_names.size, \
            "There must be a single name per label."
        if self.spp_mat is not None:
            return self.spp_mat
        img_names = np.array(img_names)
        female_idxs = labels_array == FEMALES_LABEL
        male_idxs = labels_array == MALES_LABEL

        female_masks = masks_array[female_idxs, :]
        self.female_img_names = img_names[female_idxs]
        male_masks = masks_array[male_idxs, :]
        self.male_img_names = img_names[male_idxs]

        n_fem = female_masks.shape[0]
        n_mal = male_masks.shape[0]

        self.spp_mat = np.zeros((n_fem, n_mal), dtype=float)
        for f_idx, m_idx in product(range(n_fem), range(n_mal)):
            cur_f = female_masks[f_idx, :]
            cur_m = male_masks[m_idx, :]
            self.spp_mat[f_idx, m_idx] = scoring_fn(cur_f, cur_m)
        
        # Save the arrays in the corresponding folder
        self._save_data()
        
        return self.spp_mat

    def get_sub_spp_mat(self, img_names):
        is_in_cur_females = np.array(
            [name in img_names for name in self.female_img_names]
        )
        is_in_cur_males = np.array(
            [name in img_names for name in self.male_img_names]
        )
        spp_mat = self.spp_mat[
            np.ix_(is_in_cur_females, is_in_cur_males)
        ].copy()

        return spp_mat


    def generate_pairs(self, img_names, threshold=0.1, maximize=False):
        """Generates pairs using the stored SPP Matrix. Only the images
        in img_names will be used. This is done by checking on the
        male_img_names and female_img_names attributes of the SPPMat.
        """
        # Filter the spp_mat using the img_names, so it only 
        # includes those images.
        is_in_cur_females = np.array(
            [name in img_names for name in self.female_img_names]
        )
        is_in_cur_males = np.array(
            [name in img_names for name in self.male_img_names]
        )
        spp_mat = self.spp_mat[
            np.ix_(is_in_cur_females, is_in_cur_males)
        ].copy()

        if spp_mat.shape[0] != spp_mat.shape[1]:
            raise ValueError('Data must be balanced before pairing')

        # Apply thresholds
        if maximize:
            bad_values = spp_mat[spp_mat < threshold]
            bad_values = -np.abs(np.divide(1e5, bad_values))
            spp_mat[spp_mat < threshold] = bad_values
        else:
            bad_values = spp_mat[spp_mat > threshold]
            bad_values = bad_values * 1e5
            spp_mat[spp_mat > threshold] = bad_values
        # Generate pairs
        females_pair_idx, males_pair_idx = linear_sum_assignment(
            spp_mat, maximize
        )
        # - These indexes are relative to cur_females and cur_males

        # Fix pair indexes to the order they appear in img_names
        # - Image names as they entered the SPP Array
        cur_females = self.female_img_names[is_in_cur_females]
        cur_males = self.male_img_names[is_in_cur_males]
        # - Image names as they were paired
        cur_females = cur_females[females_pair_idx]
        cur_males = cur_males[males_pair_idx]
        # - Image indexes as they appear in img_names
        female_pairs_idx = [np.where(img_names == f) for f in cur_females]
        male_pairs_idx = [np.where(img_names == m) for m in cur_males]

        return np.array([females_pair_idx, males_pair_idx])
    
    @staticmethod
    def apply_pairs(pairs, data_x, data_m):
        rescale = data_x.max() == 255
        n_pairs = pairs.shape[1]
        for i in range(n_pairs):
            cur_pair = pairs[:, i]
            cur_masks = data_m[cur_pair, :]
            cur_masks = np.any(cur_masks, axis=0) * 1
            data_m[cur_pair, :] = cur_masks
        data_x = apply_masks_to_data(data_x, data_m)
        if rescale:
            data_x *= 255

        return data_x


# def generate_spp_for_all_datasets():
#     """Single use function for generating the SPP matrixes for each
#     (both-eyes) dataset in their appropriate folders.

#     The SPP mat will be obtained for both eyes, with the left eye data
#     stacked on top of the right eye.
#     """
#     from constants import datasets_botheyes
#     from load_data import load_dataset_both_eyes
#     for dataset in datasets_botheyes:
#         all_data, males_set, females_set = load_dataset_both_eyes(dataset)
#         y_data = [all_data['left'][1], all_data['right'][1]]
#         mask_data = [all_data['left'][2], all_data['right'][2]]

