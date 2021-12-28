import numpy as np

from load_data_utils import get_labels_df, fix_labels_df, DEFAULT_ROOT_PATH


def load_dataset_from_images(dataset_name: str, root_path=DEFAULT_ROOT_PATH):
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
            img_names[i, 0][0][0].split('_')[0]
            for i in range(img_names.shape[0])
        ]
        # Check an already labeled image to check for inverse labels
        lbl_idx = 0
        labeled = iris_images_paths[lbl_idx].stem
        while labeled in unlabeled:  # Ensure labeled is labeled
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

