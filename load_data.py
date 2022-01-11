import pathlib
import platform
from pathlib import Path
from typing import Union

import numpy as np

from constants import ROOT_DATA_FOLDER, ROOT_PERI_FOLDER
from load_data_utils import get_labels_df, fix_unlabeled, \
    DEFAULT_ROOT_PATH, scale_data_by_row, partition_data, apply_masks_to_data


def load_dataset_from_images(dataset_name: str, root_path=DEFAULT_ROOT_PATH):
    """Loads the dataset directly from the images, obtaining the labels
    directly from the original GFI.txt files if available. Otherwise,
    the labels are obtained from the old datasets.
    """
    from PIL import Image
    from utils import find_n_features
    eye = dataset_name.split('_')[0]
    if dataset_name.endswith('_fixed'):
        msk_folder = 'masks'
        _dataset = dataset_name[:-6]
    else:
        msk_folder = 'old_masks'
        _dataset = dataset_name
    labels_df = get_labels_df(eye, root_path)
    dataset_folder = root_path / _dataset
    iris_images_paths = list((dataset_folder / 'iris').glob('*.bmp'))
    iris_images_paths = sorted(iris_images_paths)
    masks_folder = dataset_folder / msk_folder

    # Check that all images have a label
    unlabeled = [f.stem for f in iris_images_paths
                 if f.stem not in labels_df.filename.values]
    if unlabeled:  # Fix unlabeled
        labels_df = fix_unlabeled(
            dataset_name=dataset_name,
            iris_images_paths=iris_images_paths,
            unlabeled=unlabeled,
            labels_df=labels_df,
            labels_path=root_path
        )

    # Load images and masks
    n_features = find_n_features(dataset_name)
    n_samples = len(iris_images_paths)
    data = np.zeros((n_samples, n_features), dtype='uint8')
    labels = np.zeros(n_samples, dtype='uint8')
    masks = np.zeros((n_samples, n_features), dtype='uint8')
    for i in range(n_samples):
        cur_path = iris_images_paths[i]
        cur_label = labels_df[
            labels_df.filename == cur_path.stem].gender.values[0]
        img = np.array(Image.open(cur_path)).flatten()
        mask = np.array(Image.open(masks_folder / cur_path.name))
        mask = (mask / 255).astype('uint8').flatten()
        data[i, :] = img
        labels[i] = cur_label
        masks[i, :] = mask

    return data, labels, masks, iris_images_paths


def load_dataset_from_npz(dataset_name: str):
    """Loads the raw dataset from the npz files in the
    root data folder.
    """
    plt = platform.system()
    if plt != 'Windows':
        pathlib.WindowsPath = pathlib.PosixPath
    loaded = np.load(f'{ROOT_DATA_FOLDER}/{dataset_name}.npz',
                     allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    masks = loaded['masks']
    image_paths = loaded['image_paths']
    image_paths = [p.name for p in image_paths]
    return data, labels, masks, image_paths


def load_iris_dataset(dataset_name: str, partition: Union[int, None],
                      scale_data: bool = True, apply_masks: bool = True,
                      test_size: float = 0.3):
    """Loads the iris dataset. The dataset may be scaled to 0 and 1,
    and masks may be applied. If partition is not None, data is
    separated into train and test.

    Parameters
    ----------
    dataset_name : str
        Name of the iris dataset to load
    partition : int or None.
        Specific train/test partition to load. Used as seed. If None,
        the data and labels are returned without partitioning.
    scale_data : bool, optional
        If True, samples will be individually scaled to [0-1.0].
        This is done after applying masks (if applied). Default True.
    apply_masks : bool, optional
        If True, individual samples will be scaled to [1-255], and
        masked values will be assigned to 0. The initial scaling
        ignores masked values. Default True.
    test_size : float, optional
        Percentage of samples to use for test partition. Default 0.3.
    """
    data, labels, masks, _ = load_dataset_from_npz(dataset_name)
    # Apply masks and/or scale
    if apply_masks:
        data = apply_masks_to_data(data, masks)
    if scale_data:
        data = scale_data_by_row(data)
    if partition is None:
        return data, labels
    # Partition dataset
    return partition_data(data, labels, test_size, partition)


def load_periocular_dataset_raw(eye: str, root_path=Path('S:/'),
                                labels_path=DEFAULT_ROOT_PATH):
    """Loads the periocular iris images. This was used initially.
    Afterwards, directly loading the .npz with VGG features was
    preferred."""
    from PIL import Image
    from constants import PERIOCULAR_SHAPE
    dataset_folder = root_path / f'NUND_{eye}'
    labels_df = get_labels_df(eye, labels_path)
    iris_images_paths = list(dataset_folder.glob('*.tiff'))
    iris_images_paths = sorted(iris_images_paths)

    # Check that all images have a label
    unlabeled = [f.stem for f in iris_images_paths
                 if f.stem not in labels_df.filename.values]
    if unlabeled:  # Fix unlabeled
        labels_df = fix_unlabeled(
            dataset_name=f'{eye}_240x20_fixed',
            iris_images_paths=iris_images_paths,
            unlabeled=unlabeled,
            labels_df=labels_df,
            labels_path=labels_path
        )

    # Load images and masks
    n_features = np.product(PERIOCULAR_SHAPE)
    n_samples = len(iris_images_paths)
    data = np.zeros((n_samples, n_features), dtype='uint8')
    labels = np.zeros(n_samples, dtype='uint8')
    for i in range(n_samples):
        cur_path = iris_images_paths[i]
        cur_label = labels_df[
            labels_df.filename == cur_path.stem].gender.values[0]
        img = np.array(Image.open(cur_path)).flatten()
        data[i, :] = img
        labels[i] = cur_label

    return data, labels, iris_images_paths


def load_peri_dataset_from_npz(eye: str):
    """Loads the raw periocular dataset from the npz files in the
    root data folder.
    """
    loaded = np.load(f'{ROOT_PERI_FOLDER}/{eye}.npz',
                     allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    return data, labels


def load_dataset_both_eyes(dataset_name: str, apply_masks=True,
                           scale_data=True):
    eyes = ('left', 'right')
    if dataset_name.startswith('left') or dataset_name.startswith('right'):
        dataset_name = '_'.join(dataset_name.split('_')[1:])
    # Get all data and subject IDs
    males_set = set()
    females_set = set()
    all_data = {}
    for eye in eyes:
        cur_dataset = eye + '_' + dataset_name
        data, labels, masks, paths = load_dataset_from_npz(cur_dataset)
        if apply_masks:
            data = apply_masks_to_data(data, masks)
        if scale_data:
            data = scale_data_by_row(data)
        all_data[eye] = [data, labels, masks, paths]
        ids = np.array([p.split('d')[0] for p in paths])
        males_set.update(set(ids[labels == 0]))
        females_set.update(set(ids[labels == 1]))
    return all_data, males_set, females_set

