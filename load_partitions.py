from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def load_partitions_full(dataset_name: str, partition: int, mask_value: float,
                         test_proportion: float, scale_dataset: bool):
    np.random.seed(partition)
    # Load data
    root_path = Path.cwd() / 'data'
    data_mat = loadmat(str(root_path / (dataset_name + '.mat')))
    # Unpack .mat
    data_array = data_mat['dataArray']
    label_array = data_mat['labelArray']
    mask_array = data_mat['maskArray']
    images_list = data_mat['imagesList']
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
    # Load data
    root_path = Path.cwd() / 'data'
    idx_path = Path.cwd() / 'partIdx'
    data_mat = loadmat(str(root_path / (dataset_name + '.mat')))
    idx_mat = loadmat(str(idx_path / dataset_name / (str(partition) + '.mat')))
    # Unpack .mat
    data_array = data_mat['dataArray']
    label_array = data_mat['labelArray'][:, 0]  # Convert to 1-D
    mask_array = data_mat['maskArray']
    images_list = data_mat['imagesList'][:, 0]  # Convert to 1-D
    idx_train = idx_mat['idxTrain'][0] - 1
    idx_test = idx_mat['idxTest'][0] - 1
    # (Optionally) scale and apply mask
    if scale_dataset:
        data_array = data_array / 255
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


def load_partitions_irisbee(dataset_eye: str, partition,
                            test_proportion: float, scale_dataset: bool):
    np.random.seed(partition)
    # Load data
    root_path = Path.cwd() / 'data'
    data_mat = loadmat(str(root_path / (dataset_eye + '_240x20_irisbee.mat')))
    # Unpack .mat
    data_array = data_mat['dataArray']
    label_array = data_mat['labelArray']
    mask_array = data_mat['maskArray']
    images_list = data_mat['imagesList']
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
