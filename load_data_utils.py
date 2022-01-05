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


def fix_labels_df(images: List[str], classes: List[int], eye: str,
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