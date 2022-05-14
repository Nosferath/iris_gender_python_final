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


def balance_partition(data_x, data_y, data_m=None, data_n=None,
                      raise_error=False):
    """Balances the partition, ensuring same number of samples
    per class. If unbalanced, removes the last examples until
    balanced. This function assumes two classes. If raise_error is True,
    an error is raised if an imbalance is found, instead of balancing.
    """
    n_samples = data_x.shape[0]
    if data_y.size == n_samples:  # Regular class labels
        one_hot = False
    else:  # One hot labels
        one_hot = True
        data_y = data_y.argmax(axis=1)

    n_per_class = [np.count_nonzero(data_y == 0),
                   np.count_nonzero(data_y == 1)]
    if n_per_class[0] == n_per_class[1]:
        if one_hot:
            from vgg_utils import labels_to_onehot
            data_y = labels_to_onehot(data_y)
        to_return = [data_x, data_y]
        if data_m is not None:
            to_return.append(data_m)
        if data_n is not None:
            to_return.append(data_n)
        return to_return
    if raise_error:
        raise Exception('Unbalanced data found with raise_error flag.')
    highest = np.argmax(n_per_class)
    delta = abs(n_per_class[0] - n_per_class[1])
    locations = (data_y == highest).nonzero()[0]
    to_remove = locations[-delta:]
    to_include = np.ones(n_samples, dtype=bool)
    to_include[to_remove] = False
    data_x = data_x[to_include, :]
    data_y = data_y[to_include]
    n_per_class = [(data_y == 0).size, (data_y == 1).size]
    if n_per_class[0] != n_per_class[1]:  # Assertion. Should not happen.
        raise RuntimeError('Balance was not achieved')
    if one_hot:
        from vgg_utils import labels_to_onehot
        data_y = labels_to_onehot(data_y)
    to_return = [data_x, data_y]
    if data_m is not None:
        to_return.append(data_m[to_include, :])
    if data_n is not None:
        to_return.append(np.array(data_n)[to_include])
    return to_return


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


def permute_partitions(train_x, train_y, test_x, test_y, rng):
    train_idx = rng.permutation(len(train_y))
    train_x = train_x[train_idx, :]
    train_y = train_y[train_idx]
    test_idx = rng.permutation(len(test_y))
    test_x = test_x[test_idx, :]
    test_y = test_y[test_idx]
    return train_x, train_y, test_x, test_y


def post_partition_processing_both_eyes(
    train_x,
    train_y,
    test_x,
    test_y,
    rng
):
    """Perform partition tasks after splitting using IDs."""
    # Balance partitions
    train_x, train_y = balance_partition(train_x, train_y, raise_error=True)
    test_x, test_y = balance_partition(test_x, test_y, raise_error=True)
    # Permute partitions
    train_x, train_y, test_x, test_y = permute_partitions(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, rng=rng
    )

    return train_x, train_y, test_x, test_y


def post_partition_processing_pairs_both_eyes(
    train_data: dict, test_data: dict, rng, pairs_threshold=0.1,
    bad_bins_to_remove=0, generate_visualizations=False,
    dataset_name="no_dataset"
):
    from mask_pairs import generate_pairs, apply_pairs, remove_pairs
    # Balance partitions
    train_x, train_y, train_m, train_n = balance_partition(**train_data)
    test_x, test_y, test_m, test_n = balance_partition(**test_data)
    # Pair generation
    train_pairs, train_values = generate_pairs(train_y, train_m,
                                               threshold=pairs_threshold)
    # Generate visualizations
    if generate_visualizations:
        from results_processing import save_pairs_visualizations
        visualization_folder = 'experiments/mask_pairs/visualizations_stacked/'
        save_pairs_visualizations(train_pairs, train_x, train_m,
                                  visualization_folder + f'/{dataset_name}',
                                  to_visualize=list(
                                      range(train_pairs.shape[1])),
                                  pair_scores=train_values)
    # Apply pairs (includes scaling)
    train_x = apply_pairs(train_pairs, train_x, train_m)
    if bad_bins_to_remove:
        print('[INFO] Removing bad pairs')
        threshold = 0.11 - bad_bins_to_remove / 100
        train_x, train_y = remove_pairs(
            train_x, train_y, train_pairs, train_values, threshold=threshold)
    # Permute partitions
    train_x, train_y, test_x, test_y = permute_partitions(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, rng=rng
    )
    return train_x, train_y, test_x, test_y


def partition_both_eyes(all_data: dict, males_set: set, females_set: set,
                        test_size: float, partition: int, apply_pairs=False,
                        pairs_threshold=0.1, bad_bins_to_remove=0,
                        dataset_name=None, get_all_data_pairs=False):
    """Partition the data from both eyes ensuring both eyes of the same
    subject stay in the same partition. This is done by splitting the
    subject IDs into train and test randomly (males and females separa-
    tely to ensure balanced partitions)

    If apply_pairs is true, mask pairs are generated and applied.

    get_all_data_pairs was added for testing mask pairs.
    """
    rng = np.random.default_rng(seed=partition)
    eyes = ('left', 'right')

    # Split IDs into train and test
    males = np.array(list(males_set))
    females = np.array(list(females_set))
    n_males = len(males)
    n_females = len(females)
    test_males = rng.choice(
        males, int(test_size * n_males), replace=False
    )
    test_females = rng.choice(
        females, int(test_size * n_females), replace=False
    )

    test_ids = np.hstack([test_males, test_females])
    # Split data into partitions
    elements = ('data', 'labels', 'masks', 'names')
    train_images = {v: [] for v in elements}
    test_images = {v: [] for v in elements}
    for eye in eyes:
        cur_data = all_data[eye]
        ids = np.array([p.split('d')[0] for p in cur_data[3]])
        for i in range(len(ids)):
            if ids[i] in test_ids:
                for elem_idx, elem in enumerate(elements):
                    test_images[elem].append(cur_data[elem_idx][i])
            else:
                for elem_idx, elem in enumerate(elements):
                    train_images[elem].append(cur_data[elem_idx][i])

    train_x = np.array(train_images['data'])
    train_y = np.array(train_images['labels'])
    train_m = np.array(train_images['masks'])
    train_n = np.array(train_images['names'])
    test_x = np.array(test_images['data'])
    test_y = np.array(test_images['labels'])
    test_m = np.array(test_images['masks'])
    test_n = np.array(test_images['names'])

    if apply_pairs:
        # This has to happen between scaling and permuting, thus, needs to
        # be done separately from the regular post-partition processing.
        train_data = {
            "data_x": train_x,
            "data_y": train_y,
            "data_m": train_m,
            "data_n": train_n
        }
        test_data = {
            "data_x": test_x,
            "data_y": test_y,
            "data_m": test_m,
            "data_n": test_n
        }
        if get_all_data_pairs:
            return train_data, test_data
        train_x, train_y, test_x, test_y = \
            post_partition_processing_pairs_both_eyes(
                train_data, test_data, rng, dataset_name=dataset_name,
                pairs_threshold=pairs_threshold,
                bad_bins_to_remove=bad_bins_to_remove
            )

    else:
        # Perform post-partition tasks (balancing and permuting)
        train_x, train_y, test_x, test_y = post_partition_processing_both_eyes(
            train_x, train_y, test_x, test_y, rng
        )

    return train_x, train_y, test_x, test_y
