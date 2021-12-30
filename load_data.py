import numpy as np

from constants import ROOT_DATA_FOLDER
from load_data_utils import get_labels_df, fix_labels_df, DEFAULT_ROOT_PATH, \
    scale_data_by_row


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
        labels_tofix = []
        for ul in unlabeled:
            arr_idx = img_names.index(ul)
            old_label = int(label_arr[arr_idx])
            if invert:
                old_label = int(not bool(old_label))
            labels_tofix.append(old_label)
        fix_labels_df(unlabeled, labels_tofix, eye, root_path)
        labels_df = get_labels_df(eye, root_path)
        del data_mat, label_arr, img_names

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
    loaded = np.load(f'{ROOT_DATA_FOLDER}/{dataset_name}.npz',
                     allow_pickle=True)
    data = loaded['data']
    labels = loaded['labels']
    masks = loaded['masks']
    image_paths = loaded['image_paths']
    return data, labels, masks, image_paths


def load_iris_dataset(dataset_name: str, partition: int,
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
    from sklearn.model_selection import train_test_split
    data, labels, masks, _ = load_dataset_from_npz(dataset_name)
    n_feats = data.shape[1]
    # Apply masks and/or scale
    if apply_masks:
        # Find medians per row
        row_meds = np.median(data, axis=1)
        mids_mask = np.tile(row_meds[:, np.newaxis], (1, n_feats))
        # Temporarily set masked values in between max and min
        data[masks == 1] = mids_mask[masks == 1]
        # Scale data
        data = scale_data_by_row(data)
        # Constrain values from 1 to 255 uints
        data = np.round(data*254 + 1).astype('uint8')
        # Set mask to 0
        data[masks == 1] = 0
    if scale_data:
        data = scale_data_by_row(data)
    if partition is None:
        return data, labels
    # Partition dataset
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=test_size, stratify=labels,
        random_state=partition
    )
    return train_x, train_y, test_x, test_y
