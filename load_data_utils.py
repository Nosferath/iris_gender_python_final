from pathlib import Path
from typing import List

import numpy as np

DEFAULT_ROOT_PATH = Path('S:/NUND_fixed_masks/')


def get_labels_df(eye: str, root_path=DEFAULT_ROOT_PATH):
    """Loads the labels for each image from the GFI.txt

    Filenames are kept without their extension.
    """
    import pandas as pd
    filename = f'List_{eye.lower()}_GFI.txt'
    df = pd.read_csv(root_path / filename, sep='\t', header=None)
    df.columns = ['filename', 'gender']
    df.loc[:, 'filename'] = df.filename.apply(lambda x: x.split('.')[0])
    return df


def fix_unlabeled(dataset_name: str, iris_images_paths: list,
                  unlabeled: list, labels_df,
                  labels_path=DEFAULT_ROOT_PATH):
    """Finds the labels of images that do not have their
    labels in the GFI.txt file.

    The labels are taken from previously found labels, in old
    datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    iris_images_paths : list
        Sorted list of the paths of the images that are being
        loaded. Obtained using glob().
    unlabeled : list
        List of file stems that were not found in GFI.txt and
        thus are not present in labels_df
    labels_df : pd.DataFrame
        DataFrame with labels, obtained using get_labels_df
    labels_path : pathlib Path, optional
        Path where the GFI.txt file is located
    """
    from scipy.io import loadmat
    data_mat = loadmat(f'_old_data/{dataset_name}.mat')
    label_arr = data_mat['labelArray'][:, 0]
    img_names = data_mat['imagesList']
    img_names = [
        img_names[i, 0][0][0].split('_')[0].split('.')[0]
        for i in range(img_names.shape[0])
    ]
    # Check an already labeled image to check for inverse labels
    lbl_idx = 0
    labeled = iris_images_paths[lbl_idx].stem
    while labeled in unlabeled or labeled not in img_names:
        # Ensure labeled is labeled
        lbl_idx += 1
        labeled = iris_images_paths[lbl_idx].stem
    df_label = labels_df[labels_df.filename == labeled].gender.values[0]
    arr_idx = img_names.index(labeled)
    old_label = label_arr[arr_idx]
    if int(old_label) == int(df_label):
        invert = False
    else:
        invert = True
    # Fix unlabeled
    eye = dataset_name.split('_')[0]
    labels_tofix = []
    for ul in unlabeled:
        arr_idx = img_names.index(ul)
        old_label = int(label_arr[arr_idx])
        if invert:
            old_label = int(not bool(old_label))
        labels_tofix.append(old_label)
    fix_labels_txt(unlabeled, labels_tofix, eye, labels_path)
    labels_df = get_labels_df(eye, labels_path)
    del data_mat, label_arr, img_names

    return labels_df


def fix_labels_txt(images: List[str], classes: List[int], eye: str,
                   root_path=DEFAULT_ROOT_PATH):
    """Adds the missing labels to the GFI.txt.
    DOES NOT CHECK FOR DUPLICATES.

    Parameters
    ----------
    images : List of strings
        Contains the STEMS of the images' filenames
    classes : List of ints
        Contains the classes of the images, as 0s or 1s
    eye : str
        Eye of the dataset. Either 'left' or 'right'
    root_path : Pathlib Path, optional
        Path where the GFI.txt file is located.
    """
    with open(root_path / f'List_{eye.lower()}_GFI.txt', 'a') as f:
        for img, c in zip(images, classes):
            print(f'Adding file {img} with label {c} to labels')
            f.write(f'\n{img}.tiff\t{c}')


def save_raw_dataset(data: np.ndarray, labels: np.ndarray, masks: np.ndarray,
                     image_paths: List, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out_file,
        data=data,
        labels=labels,
        masks=masks,
        image_paths=np.array(image_paths)
    )


def scale_data_by_row(data: np.array):
    n_feats = data.shape[1]
    row_mins = data.min(axis=1)
    row_maxs = data.max(axis=1)
    tile_mins = np.tile(row_mins[:, np.newaxis], (1, n_feats))
    tile_maxs = np.tile(row_maxs[:, np.newaxis], (1, n_feats))
    # Scale rows using these values
    data = np.divide(data - tile_mins, tile_maxs - tile_mins)
    return data


def balance_partition(data_x, data_y):
    """Balances the partition, ensuring same number of samples
    per class. If unbalanced, removes the last examples until
    balanced.
    """
    n_samples = data_x.shape[0]
    n_per_class = [(data_y == 0).size, (data_y == 1).size]
    if n_per_class[0] == n_per_class[1]:
        return data_x, data_y
    highest = np.argmax(n_per_class)
    delta = abs(n_per_class[0] - n_per_class[1])
    locations = (data_y == highest).nonzero()[0]
    to_remove = locations[-delta:]
    to_include = np.ones(n_samples, dtype=bool)
    to_include[to_remove] = False
    return data_x[to_include, :], data_y[to_include]


def partition_data(data, labels, test_size: float, partition: int):
    """Partitions and balances the data"""
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=test_size, stratify=labels,
        random_state=partition
    )
    train_x, train_y = balance_partition(train_x, train_y)
    test_x, test_y = balance_partition(test_x, test_y)
    return train_x, train_y, test_x, test_y


def apply_masks_to_data(data, masks):
    """Applies the masks to the data. For this, the non-masked values
    are scaled to 1-255 (each sample is scaled individually), and masks
    are applied on value 0.
    """
    data = data.copy()
    n_feats = data.shape[1]
    # Find medians per row
    row_meds = np.median(data, axis=1)
    mids_mask = np.tile(row_meds[:, np.newaxis], (1, n_feats))
    # Temporarily set masked values in between max and min
    data[masks == 1] = mids_mask[masks == 1]
    # Scale data
    data = scale_data_by_row(data)
    # Constrain values from 1 to 255 uints
    data = np.round(data * 254 + 1).astype('uint8')
    # Set mask to 0
    data[masks == 1] = 0
    return data


def partition_both_eyes(all_data: dict, males_set: set, females_set: set,
                        test_size: float, partition: int):
    rng = np.random.default_rng(partition)
    eyes = ('left', 'right')
    # Split IDs into train and test
    males = np.array(list(males_set))
    females = np.array(list(females_set))
    n_males = len(males)
    n_females = len(females)
    test_males = rng.choice(
        males, np.int(test_size * n_males), replace=False
    )
    test_females = rng.choice(
        females, np.int(test_size * n_females), replace=False
    )
    test_ids = np.hstack([test_males, test_females])
    # Split data into partitions
    train_images = {v: [] for v in ('data', 'labels')}
    test_images = {v: [] for v in ('data', 'labels')}
    for eye in eyes:
        data, labels, masks, paths = all_data[eye]
        ids = np.array([p.split('d')[0] for p in paths])
        for i in range(len(ids)):
            if ids[i] in test_ids:
                test_images['data'].append(data[i, :])
                test_images['labels'].append(labels[i])
            else:
                train_images['data'].append(data[i, :])
                train_images['labels'].append(labels[i])
    train_x = np.array(train_images['data'])
    train_y = np.array(train_images['labels'])
    test_x = np.array(test_images['data'])
    test_y = np.array(test_images['labels'])
    # Balance partitions
    train_x, train_y = balance_partition(train_x, train_y)
    test_x, test_y = balance_partition(test_x, test_y)
    # Permutate partitions
    train_idx = rng.permutation(len(train_y))
    train_x = train_x[train_idx, :]
    train_y = train_y[train_idx]
    test_idx = rng.permutation(len(test_y))
    test_x = test_x[test_idx, :]
    test_y = test_y[test_idx]

    return train_x, train_y, test_x, test_y
