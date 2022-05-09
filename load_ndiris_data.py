from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from constants import MALES_LABEL, FEMALES_LABEL

# Folder structure is
# ROOT_PATH/SIZE/SENSOR/[left,right]/Normalized[Images,Masks]/*.bmp
ROOT_PATH = Path().cwd().parent
NDIRIS_240x20 = ROOT_PATH / 'NDIRIS_240x20'
NDIRIS_240x20_SHAPE = (20, 240)
NDIRIS_240x40 = ROOT_PATH / 'NDIRIS_240x40'
NDIRIS_240x40_SHAPE = (40, 240)
NDIRIS_DATA_FOLDER = 'data_ndiris'
MASKS_FOLDER = 'NormalizedMasks'
SENSOR_LG4000 = 'LG4000'
SENSOR_LG2000 = 'LG2200'
SENSORS = (SENSOR_LG4000, SENSOR_LG2000)


def get_dataset_df_raw(root_folder: Union[str, Path],
                       sensor: str = SENSOR_LG4000) -> pd.DataFrame:
    """Generates a DataFrame with the dataset information by checking
    the actual image files.
    """
    root_folder = Path(root_folder)
    if root_folder.name.startswith('NDIRIS_240'):
        images_list = list(
            root_folder.glob(f'{sensor}/*/NormalizedImages/*.bmp'))
    else:  # ROOT_PATH
        images_list = list(
            root_folder.glob('*/{sensor}/*/NormalizedImages/*.bmp'))
    if len(images_list) == 0:
        raise FileNotFoundError('No files were found')
    df = pd.DataFrame({'path': images_list})
    df['size'] = df['path'].apply(
        lambda x: list(x.parents)[3].name.split('_')[-1])
    df['sensor'] = df['path'].apply(
        lambda x: list(x.parents)[2].name)
    df['eye'] = df['path'].apply(
        lambda x: list(x.parents)[1].name)
    df['id_number'] = df['path'].apply(
        lambda x: int(x.stem.split('d')[0]))

    older_df = pd.read_csv(NDIRIS_240x20 / 'dataframe_old.csv')
    labels_csvs_list = list(ROOT_PATH.glob('*/*/*/labels.csv'))
    labels_dfs = [pd.read_csv(f) for f in labels_csvs_list]
    labels_df = pd.concat(labels_dfs)

    def find_gender_in_older_df(img_path):
        img_name = img_path.stem.split('_')[0]
        labels = labels_df[labels_df.img_id == img_name].gender.values
        if len(labels) != 0:
            # Reverse of what the other dataset uses
            gender = MALES_LABEL if labels[0] == 0 else FEMALES_LABEL
            return gender
        img_id = int(img_path.stem.split('d')[0])
        labels_older = older_df[older_df.id_number == img_id].gender.dropna()
        gender = FEMALES_LABEL if labels_older.values[0] == "Female" else \
            MALES_LABEL
        return gender

    df['gender'] = df.path.apply(find_gender_in_older_df)
    return df


def get_dataset_df(root_folder: Union[str, Path],
                   sensor: str = SENSOR_LG4000) -> pd.DataFrame:
    if root_folder == ROOT_PATH:
        return get_dataset_df_raw(root_folder, sensor)
    root_folder = Path(root_folder)
    size = root_folder.name.split('_')[-1]
    df = pd.read_csv(NDIRIS_DATA_FOLDER + f'/{size}_{sensor}.csv', index_col=0)
    df.path = df.path.apply(lambda x: Path(x))
    return df


def dataset_summary(out_folder='experiments/ndiris_summary/'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = get_dataset_df(ROOT_PATH)
    df.gender = df.gender.apply(lambda x: "Female" if x == 0 else "Male")

    # Statistics
    n_subjects = len(df.id_number.unique())
    print(f'Number of subjects: {n_subjects}')
    male_subjects = len(df[df.gender == "Male"].id_number.unique())
    print(f'\tMale subjects: {male_subjects}')
    female_subjects = len(df[df.gender == "Female"].id_number.unique())
    print(f'\tFemale subjects: {female_subjects}\n')

    images_per_subject = df.groupby('id_number').count().path.describe()
    print(f'Images per subject: \n{images_per_subject}\n')
    male_images = len(df[df.gender == 'Male'])
    print(f'Male images: \n{male_images}\n')
    female_images = len(df[df.gender == 'Female'])
    print(f'Female images: \n{female_images}\n')
    images_per_sensor = df.groupby('sensor').count().path
    print(f'Images per sensor: \n{images_per_sensor}\n')

    with sns.plotting_context('talk'):
        gender_hist = df.groupby('id_number').gender.describe()
        sns.histplot(gender_hist, x='count', hue='top', binwidth=10)
        plt.legend(title='gender', loc='upper right',
                   labels=['Male', 'Female'])
        plt.xlabel('Number of images')
        plt.title('Number of images per subject, NDIris Dataset')
        plt.tight_layout()
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_folder / 'images_histogram.png')

    return df


def load_dataset_raw(size=None, sensor=SENSOR_LG4000, df=None):
    """Loads the NDIRIS dataset from the images"""
    import cv2

    assert size in ['240x20', '240x40', None], \
        'Size must be either "240x20", "240x40" or None'
    assert (size is not None and df is None) or \
        (size is None and df is not None), \
        'Either size or df must be None, not both'

    if df is None:
        root_folder = NDIRIS_240x20 if size == '240x20' else NDIRIS_240x40
        df = get_dataset_df(root_folder=root_folder, sensor=sensor)
    else:
        size = df['size'].iloc[0]

    shape = NDIRIS_240x20_SHAPE if size == '240x20' \
        else NDIRIS_240x40_SHAPE
    n_features = np.prod(shape)
    n_images = len(df)
    x_array = np.zeros((n_images, n_features))
    y_array = np.zeros(n_images)
    m_array = np.zeros((n_images, n_features))
    l_array = np.zeros(n_images, dtype='object')
    for idx, image_row in df.reset_index().iterrows():
        image_path = image_row.path
        image: np.array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        x_array[idx, :] = image.flatten()
        y_array[idx] = image_row.gender
        mask_path = image_path.parent.parent / MASKS_FOLDER
        mask_path = mask_path / (image_path.name.split('_')[0] + '_mano.bmp')
        mask: np.array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        m_array[idx, :] = mask.flatten()
        l_array[idx] = image_path

    return x_array, y_array, m_array, l_array, df


def train_test_split(df, test_size, partition):
    # Limit to 20 images per subject (10 per eye)
    df = df.groupby(['id_number', 'eye']).head(10)

    # Separate males and females into train and test
    rng = np.random.default_rng(seed=partition)

    males = df[df.gender == MALES_LABEL]
    male_ids = np.sort(males.id_number.unique())
    n_males = males.id_number.nunique()
    females = df[df.gender == FEMALES_LABEL]
    female_ids = np.sort(females.id_number.unique())
    n_females = females.id_number.nunique()
    test_males = rng.choice(
        male_ids, int(n_males * test_size), replace=False
    )
    test_females = rng.choice(
        female_ids, int(n_females * test_size), replace=False
    )
    test_ids = np.hstack([test_males, test_females])
    test_images = df[df.id_number.isin(test_ids)]
    train_images = df[~df.id_number.isin(test_ids)]

    # Balance number of male and female images
    def balance_df(df_to_balance):
        df_males = df_to_balance[df_to_balance.gender == MALES_LABEL]
        df_females = df_to_balance[df_to_balance.gender == FEMALES_LABEL]
        n_balance = min(len(df_males), len(df_females))
        df_males = df_males.groupby('eye').head(int(n_balance / 2))
        df_females = df_females.groupby('eye').head(int(n_balance / 2))

        return pd.concat([df_males, df_females])

    train_images = balance_df(train_images)
    test_images = balance_df(test_images)

    return train_images, test_images


def split_from_partition_df(x_array, y_array, m_array, l_array, part_df):
    """Extracts the images in the part_df from the arrays. Useful for
    loading and splitting pre-processed VGG data.
    """
    selected = np.isin(l_array, part_df.path.values)
    x_array = x_array[selected, :]
    y_array = y_array[selected]
    to_return = [x_array, y_array]
    if m_array is not None:
        m_array = m_array[selected, :]
        to_return.append(m_array)
    l_array = l_array[selected]
    to_return.append(l_array)

    return to_return


def apply_masks(x_array, y_array, m_array, use_pairs: bool):
    # Apply masks (include scaling)
    # Change masks from 0-mask/255-nomask to 1-mask/0-nomask
    if m_array.max() == 255:
        masked = m_array == 0
        m_array[masked] = 1
        m_array[~masked] = 0
    if use_pairs:
        from mask_pairs import generate_pairs, apply_pairs

        pairs, pair_scores = generate_pairs(y_array, m_array)
        x_array = apply_pairs(pairs, x_array, m_array)

    else:
        from load_data_utils import apply_masks_to_data

        x_array = apply_masks_to_data(x_array, m_array)

    return x_array


def permute_images(train_x, train_y, train_m, train_l,
                   test_x, test_y, test_m, test_l):
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(train_y))
    train_x = train_x[train_idx, :]
    train_y = train_y[train_idx]
    to_return = [train_x, train_y]
    if train_m is not None:
        train_m = train_m[train_idx, :]
        to_return.append(train_m)
    if train_l is not None:
        train_l = train_l[train_idx]
        to_return.append(train_l)
    test_idx = rng.permutation(len(test_y))
    test_x = test_x[test_idx, :]
    to_return.append(test_x)
    test_y = test_y[test_idx]
    to_return.append(test_y)
    if test_m is not None:
        test_m = test_m[test_idx, :]
        to_return.append(test_m)
    if test_l is not None:
        test_l = test_l[test_idx]
        to_return.append(test_l)

    return to_return


def load_ndiris_dataset(size: str, partition, use_pairs: bool,
                        test_size=0.2, sensor=SENSOR_LG4000):
    """Full pipeline for loading NDIris dataset"""
    assert size in ['240x20', '240x40'], \
        'Size must be either "240x20" or "240x40"'
    root_folder = NDIRIS_240x20 if size == '240x20' else NDIRIS_240x40
    # Load DF
    df = get_dataset_df(root_folder, sensor)
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size, partition)
    # Load images
    train_x, train_y, train_m, train_l, _ = load_dataset_raw(sensor=sensor,
                                                             df=train_df)
    test_x, test_y, test_m, test_l, _ = load_dataset_raw(sensor=sensor,
                                                         df=test_df)
    # Apply masks
    train_x = apply_masks(train_x, train_y, train_m, use_pairs)
    test_x = apply_masks(test_x, test_y, test_m, use_pairs=False)

    # Permute images
    train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l = \
        permute_images(train_x, train_y, train_m, train_l,
                       test_x, test_y, test_m, test_l)

    return train_x, train_y, train_m, train_l, test_x, test_y, test_m, test_l
