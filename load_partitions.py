from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from constants import CMIM_GROUPS, ROOT_DATA_FOLDER
from pairs import prepare_pairs_indexes, load_pairs_array
from standard_masks import generate_standard_masks, apply_std_mask
from utils import find_dataset_shape


def scale_x_arr(x_arr: np.ndarray, mask_array: np.ndarray,
                mask_value: float = 0):
    """Scales each individual sample of the x_arr, such that all non-
    masked values are between 1/255 and 1.0, leaving the value 0 for
    masks.
    """
    temp_mask = np.repeat(
        np.median(x_arr, axis=1).reshape(-1, 1),
        x_arr.shape[1], axis=1
    )
    x_arr[mask_array == 1] = temp_mask[mask_array == 1]
    x_arr[x_arr >= 255] = 254
    max_values = x_arr.max(axis=1)
    max_values = np.repeat(max_values.reshape(-1, 1),
                           x_arr.shape[1],
                           axis=1)
    min_values = x_arr.min(axis=1)
    min_values = np.repeat(min_values.reshape(-1, 1),
                           x_arr.shape[1],
                           axis=1)
    x_arr = np.divide(x_arr - min_values, max_values - min_values)
    x_arr = (254.0 * x_arr + 1) / 255.0
    x_arr[mask_array == 1] = mask_value
    return x_arr


def load_raw_dataset(dataset_name: str, root_folder=ROOT_DATA_FOLDER):
    """Loads a full (non-partitioned) dataset from a .mat file."""
    root_folder = Path.cwd() / root_folder
    data_mat = loadmat(str(root_folder / (dataset_name + '.mat')))
    data_array = data_mat['dataArray']
    label_array = data_mat['labelArray'][:, 0]  # Convert to 1-D
    mask_array = data_mat['maskArray']
    images_list = data_mat['imagesList']
    images_list = [
        images_list[i, 0][0][0] for i in range(images_list.shape[0])
    ]
    images_list = np.array(list(map(lambda x: x.split('_')[0], images_list)))
    return data_array, label_array, mask_array, images_list


def load_partitions_full(dataset_name: str, partition: int, mask_value: float,
                         test_proportion: float, scale_dataset: bool,
                         root_folder=ROOT_DATA_FOLDER):
    """This version of load_partitions generates its own balanced partitions.
    Not used currently. Does not include newest scaling."""
    np.random.seed(partition)
    # Load data
    data_array, label_array, mask_array, images_list = load_raw_dataset(
        dataset_name, root_folder)
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
                    scale_dataset: bool, root_folder=ROOT_DATA_FOLDER):
    """Loads a partitioned dataset. Uses .mat files provided by MATLAB.
    Uses new scaling for 0-masking.
    """
    # Load data
    data_array, label_array, mask_array, images_list = load_raw_dataset(
        dataset_name, root_folder)
    idx_path = Path.cwd() / 'partIdx'
    idx_mat = loadmat(str(idx_path / dataset_name / (str(partition) + '.mat')))
    # Unpack .mat
    idx_train = idx_mat['idxTrain'][0] - 1
    idx_test = idx_mat['idxTest'][0] - 1
    # (Optionally) scale and apply mask
    if scale_dataset:
        data_array = scale_x_arr(data_array, mask_array, mask_value)
    else:
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
                               pair_method: str, exclude: int, use_max: bool,
                               root_folder=ROOT_DATA_FOLDER):
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
        load_partitions(dataset_name, partition, mask_value, scale_dataset,
                        root_folder)
    if exclude and not pair_method:
        raise ValueError('If exclude is defined, pair_method must also be.')
    if pair_method:
        pairs = load_pairs_array(dataset_name=dataset_name,
                                 pair_method=pair_method,
                                 partition=partition)
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
        if scale_dataset:
            train_x = scale_x_arr(train_x, train_m, mask_value)
            test_x = scale_x_arr(test_x, test_m, mask_value)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_pairs(dataset_name: str, partition: int, mask_value: float,
                          scale_dataset: bool, pair_method: str,
                          root_folder=ROOT_DATA_FOLDER):
    """Loads the partition. If pair_method is False, it is identical to
    regular load_partitions. Otherwise, pair_method should be a string
    for a pair type (a folder in the pairs folder). The pairs from that
    method will have their masks standardized. (Step 2, 6 and 7)
    """
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs_base(dataset_name, partition, mask_value,
                                   scale_dataset, pair_method, 0, False,
                                   root_folder)
    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def apply_cmim_to_partition(train_x: np.ndarray, test_x: np.ndarray,
                            dataset_name: str, pair_method: str, n_cmim: int):
    """Applies feature selection to the x arrays of the partition."""
    from cmim import load_cmim_array
    if n_cmim < 0:
        raise ValueError('n_cmim must be greater than 0')
    if n_cmim:
        cmim_array = load_cmim_array(dataset_name, pair_method)
        feats_per_group = int(train_x.shape[1] / CMIM_GROUPS)
        feats_total = n_cmim * feats_per_group
        selected = cmim_array[:feats_total]
        train_x = train_x[:, selected]
        test_x = test_x[:, selected]
    return train_x, test_x


def load_partitions_cmim(dataset_name: str, partition: int, mask_value: float,
                         scale_dataset: bool, pair_method: str, n_cmim: int,
                         root_folder=ROOT_DATA_FOLDER):
    """Loads the partition. It keeps only the n_cmim groups of most
    important features according to CMIM, on the train_x and test_x
    arrays.
    """
    if n_cmim < 0:
        raise ValueError('n_cmim must be greater than 0')

    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs(dataset_name, partition, mask_value,
                              scale_dataset, pair_method, root_folder)

    train_x, test_x = apply_cmim_to_partition(train_x, test_x, dataset_name,
                                              pair_method, n_cmim)

    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_cmim_mod(dataset_name: str, partition: int,
                             mask_value: float, scale_dataset: bool,
                             pair_method: str, n_cmim: int):
    """Loads the partition, but applies an artificial modification to
    the dataset, turning a certain area white for female samples, and
    black for male samples.
    """
    from cmim import artificial_mod_dataset
    if n_cmim < 0:
        raise ValueError('n_cmim must be greater than 0')

    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs(dataset_name, partition, mask_value,
                              scale_dataset, pair_method)
    train_x, test_x = artificial_mod_dataset(train_x, train_y, test_x, test_y)
    train_x, test_x = apply_cmim_to_partition(train_x, test_x, dataset_name,
                                              pair_method, n_cmim)

    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l


def load_partitions_pairs_excl(dataset_name: str, partition: int,
                               mask_value: float, scale_dataset: bool,
                               pair_method: str, exclude: int, use_max: bool,
                               root_folder=ROOT_DATA_FOLDER):
    """Loads the partition, performs pairing and excludes a number of
    pairs (worst pairs for the pair_method); exclude is the number of
    pairs to exclude. pair_method should be a string for a pair type (a
    folder in the pairs folder). The pairs from that method will have
    their masks standardized. (Step 2, 6 and 7)

    If pair_method or exclude are not to be used, they must be set to
    False/0. If exclude is to be used, use_max must be set accordingly.
    If exclude is defined, pair_method must also be defined.
    """
    # FIXME this should probably also return the modified pairs array
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        load_partitions_pairs_base(dataset_name, partition, mask_value,
                                   scale_dataset, pair_method, exclude,
                                   use_max, root_folder)
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


def generate_mask_visualization(dataset_name: str, pairs: str, partition=1):
    """Generates a grayscale visualization of the masks of the dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use
    pairs : str or None/False
        Set to None/False if pairs are not to be used. Otherwise, set to the
        pairing method name.
    partition : int
        Train partition to use. Default 1.
    """
    _, _, train_m, _, _, _, _, _ = load_partitions_pairs(
        dataset_name, partition, 0, True, pairs)
    masks = train_m.mean(axis=0)
    masks = masks * 255
    shape = find_dataset_shape(dataset_name)
    masks = masks.reshape(shape)
    return masks.astype('uint8')


def generate_color_bg(dataset_name: str, color='black'):
    """Generates a visualization of a specific color in the shape of the
    dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Used for determining size.
    color : str
        Color to use. Default is 'magenta'. Only magenta implemented.
    """
    colors = ['black']
    if color not in colors:
        raise NotImplemented(f'Color {color} not implemented')
    shape = find_dataset_shape(dataset_name)
    if color == colors[0]:  # magenta
        ones = np.ones(shape, dtype='uint8')
        zeros = np.zeros(shape, dtype='uint8')
        output = np.concatenate([zeros[..., np.newaxis],
                                 zeros[..., np.newaxis],
                                 zeros[..., np.newaxis]], axis=2)
        return (255 * output).astype('uint8')
