from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def prepare_pairs_indexes(pairs: np.ndarray) -> np.ndarray:
    """Takes the indexes from the pairs array, turns them to ints, and
    ensures they start from 0. It has no effect if the pairs array has
    been previously prepared.
    """
    pairs = pairs[:, :2].astype(int)
    if np.min(pairs) == 1:
        pairs = pairs - 1
    return pairs


def generate_standard_masks(mask_array: np.ndarray, pairs: np.ndarray):
    """Generate the standard masks."""
    std_masks = mask_array.copy()
    n_pairs = pairs.shape[0]
    pairs = prepare_pairs_indexes(pairs)
    for i in range(n_pairs):
        cur_pairs = pairs[i, :]
        cur_masks = mask_array[cur_pairs, :]
        std_mask = np.any(cur_masks, axis=0) * 1  # CHECK
        std_masks[cur_pairs, :] = np.vstack([std_mask, std_mask])
    return std_masks


def apply_std_mask(x_array: np.ndarray, std_masks: np.ndarray,
                   mask_value: float):
    """Applies the std masks to the x array."""
    masked_x_array = x_array.copy()
    masked_x_array[std_masks == 1] = mask_value
    return masked_x_array


def load_raw_dataset(dataset_name: str):
    """Loads a full (non-partitioned) dataset from a .mat file."""
    root_folder = Path.cwd() / 'data'
    data_mat = loadmat(str(root_folder / (dataset_name + '.mat')))
    data_array = data_mat['dataArray']
    label_array = data_mat['labelArray']
    mask_array = data_mat['maskArray']
    images_list = data_mat['imagesList']
    images_list = [
        images_list[i, 0][0][0] for i in range(images_list.shape[0])
    ]
    images_list = np.array(list(map(lambda x: x.split('_')[0], images_list)))
    return data_array, label_array, mask_array, images_list


def load_partitions_full(dataset_name: str, partition: int, mask_value: float,
                         test_proportion: float, scale_dataset: bool):
    """This version of load_partitions generates its own balanced partitions.
    Not used currently. Does not include newest scaling."""
    np.random.seed(partition)
    # Load data
    data_array, label_array, mask_array, images_list = load_raw_dataset(
        dataset_name)
    # Balance and (optionally) scale dataset
    n_images = len(label_array)
    idx = np.random.permutation(n_images)
    data_array = data_array[idx, :]
    label_array = label_array[idx, 0]  # Convert to 1-D
    mask_array = mask_array[idx, :]
    images_list = images_list[idx, 0]  # Convert to 1-D
    # Find class with less examples
    n_males = np.sum(label_array == 0)
    n_females = np.sum(label_array == 1)
    if n_males != n_females:  # Balance partitions
        max_value = np.max([n_males, n_females])
        max_class = np.argmax([n_males, n_females])
        delta = abs(n_males - n_females)
        selected = np.full(n_images, True)
        max_class_select = np.full(max_value, True)
        max_class_select[:delta] = False
        selected[label_array == max_class] = max_class_select

        data_array = data_array[selected, :]
        if scale_dataset:
            data_array = data_array / 255.0
        label_array = label_array[selected]
        mask_array = mask_array[selected, :]
        images_list = images_list[selected]

        assert np.sum(label_array == 0) == np.sum(label_array == 1), \
            'Class balance did not work'
    # Apply mask
    data_array[mask_array == 1] = mask_value
    # Generate train and test
    train_x, test_x, train_y, test_y, train_m, test_m, train_l, test_l = \
        train_test_split(data_array, label_array, mask_array, images_list,
                         test_size=test_proportion, stratify=label_array,
                         random_state=partition)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions(dataset_name: str, partition: int, mask_value: float,
                    scale_dataset: bool):
    """Loads a partitioned dataset. Uses .mat files provided by MATLAB.
    Uses new scaling for 0-masking.
    """
    # Load data
    data_array, label_array, mask_array, images_list = load_raw_dataset(
        dataset_name)
    idx_path = Path.cwd() / 'partIdx'
    idx_mat = loadmat(str(idx_path / dataset_name / (str(partition) + '.mat')))
    # Unpack .mat
    idx_train = idx_mat['idxTrain'][0] - 1
    idx_test = idx_mat['idxTest'][0] - 1
    # (Optionally) scale and apply mask
    if scale_dataset:
        data_array[mask_array == 1] = 177
        data_array[data_array == 255] = 254
        max_values = data_array.max(axis=1)
        max_values = np.repeat(max_values.reshape(-1, 1), data_array.shape[1],
                               axis=1)
        min_values = data_array.min(axis=1)
        min_values = np.repeat(min_values.reshape(-1, 1), data_array.shape[1],
                               axis=1)
        data_array = np.divide(data_array - min_values,
                               max_values - min_values)
        data_array = (254.0 * data_array + 1) / 255.0
    data_array[mask_array == 1] = mask_value
    # Generate train and test
    train_x = data_array[idx_train, :]
    train_y = label_array[idx_train]
    train_m = mask_array[idx_train, :]
    train_l = images_list[idx_train]
    test_x = data_array[idx_test, :]
    test_y = label_array[idx_test]
    test_m = mask_array[idx_test, :]
    test_l = images_list[idx_test]
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_pairs_base(dataset_name: str, partition: int,
                               mask_value: float, scale_dataset: bool,
                               pair_method: str, exclude: int, use_max: bool):
    """Base function for both load_partitions_pairs and load_partitions_
    pairs_excl. If pair_method or exclude are not to be used, they must
    be set to False. If exclude is to be used, use_max must be set
    accordingly. If exclude is defined, pair_method must also be defined

    Loads the partition. If pair_method is False, it is identical to
    regular load_partitions. Otherwise, pair_method should be a string
    for a pair type (a folder in the pairs folder). The pairs from that
    method will have their masks standardized. (Step 2, 6 and 7)
    """
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions(dataset_name, partition, mask_value, scale_dataset)
    if exclude and not pair_method:
        raise ValueError('If exclude is defined, pair_method must also be.')
    if pair_method:
        pairs_path = Path.cwd() / 'pairs' / pair_method / dataset_name
        pairs_mat = loadmat(str(pairs_path / (str(partition) + '.mat')))
        pairs = pairs_mat['pairs']
        train_m = generate_standard_masks(train_m, pairs)
        train_x = apply_std_mask(train_x, train_m, mask_value)
        if exclude:
            idx_list = np.argsort(pairs[:, 2])
            if not use_max:
                idx_list = idx_list[::-1]
            # Vector of excluded indexes
            pairs = prepare_pairs_indexes(pairs)
            to_exclude = pairs[idx_list[:exclude], :].flatten()
            # Convert excluded indexes to boolean
            to_exclude_bool = np.full((len(to_exclude), 1), True)
            to_exclude_bool[to_exclude] = False
            # Apply exclusion
            train_x = train_x[to_exclude_bool, :]
            train_y = train_y[to_exclude_bool]
            train_m = train_m[to_exclude_bool, :]
            train_l = train_l[to_exclude_bool]
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_pairs(dataset_name: str, partition: int, mask_value: float,
                          scale_dataset: bool, pair_method: str):
    """Loads the partition. If pair_method is False, it is identical to
    regular load_partitions. Otherwise, pair_method should be a string
    for a pair type (a folder in the pairs folder). The pairs from that
    method will have their masks standardized. (Step 2, 6 and 7)
    """
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs_base(dataset_name, partition, mask_value,
                                   scale_dataset, pair_method, 0, False)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_pairs_excl(dataset_name: str, partition: int,
                               mask_value: float, scale_dataset: bool,
                               pair_method: str, exclude: int, use_max: bool):
    """Loads the partition, performs pairing and excludes a number of
    pairs (worst pairs for the pair_method); exclude is the number of
    pairs to exclude. pair_method should be a string for a pair type (a
    folder in the pairs folder). The pairs from that method will have
    their masks standardized. (Step 2, 6 and 7)

    If pair_method or exclude are not to be used, they must be set to
    False/0. If exclude is to be used, use_max must be set accordingly.
    If exclude is defined, pair_method must also be defined.
    """
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs_base(dataset_name, partition, mask_value,
                                   scale_dataset, pair_method, exclude,
                                   use_max)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_irisbee_full(dataset_eye: str, partition: int,
                                 test_proportion: float, scale_dataset: bool):
    """Function for loading non irisBee datasets and partitions. Generates its
    own partitions. Not used. Does not include new scaling."""
    np.random.seed(partition)
    # Load data
    dataset_name = dataset_eye + '_240x20_irisbee'
    data_array, label_array, mask_array, images_list = load_raw_dataset(
        dataset_name)
    # Random permute
    n_images = len(label_array)
    idx = np.random.permutation(n_images)
    data_array = data_array[idx, :]
    label_array = label_array[idx, 0]  # Convert to 1-D
    mask_array = mask_array[idx, :]
    images_list = images_list[idx, 0]  # Convert to 1-D
    # Scale dataset
    if scale_dataset:
        data_array = data_array / 255.0
    # Generate train and test
    train_x, test_x, train_y, test_y, train_m, test_m, train_l, test_l = \
        train_test_split(data_array, label_array, mask_array, images_list,
                         test_size=test_proportion, stratify=label_array,
                         random_state=partition)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l
