from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


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
            gender = 1 if labels[0] == 0 else 0
            return gender
        img_id = int(img_path.stem.split('d')[0])
        labels_older = older_df[older_df.id_number == img_id].gender.dropna()
        gender = 0 if labels_older.values[0] == "Female" else "Male"
        return gender

    df['gender'] = df.path.apply(find_gender_in_older_df)
    return df


def get_dataset_df(root_folder: Union[str, Path],
                   sensor: str = SENSOR_LG4000) -> pd.DataFrame:
    if root_folder == ROOT_PATH:
        return get_dataset_df_raw(root_folder, sensor)
    root_folder = Path(root_folder)
    size = root_folder.name.split('_')[-1]
    df = pd.read_csv(NDIRIS_DATA_FOLDER + f'/{size}_{sensor}.csv')
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


def load_dataset_raw(size, sensor=SENSOR_LG4000):
    import cv2

    assert size in ['240x20', '240x40'], \
        'Size must be either "240x20" or "240x40"'
    root_folder = NDIRIS_240x20 if size == '240x20' else NDIRIS_240x40
    shape = NDIRIS_240x20_SHAPE if size == '240x20' else NDIRIS_240x40_SHAPE
    n_features = np.prod(shape)

    df = get_dataset_df(root_folder=root_folder, sensor=sensor)
    n_images = len(df)
    x_array = np.zeros((n_images, n_features))
    y_array = np.zeros(n_images)
    m_array = np.zeros((n_images, n_features))
    l_array = np.zeros(n_images, dtype='object')
    for idx, image_row in df.iterrows():
        image_path = image_row.path
        image: np.array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        x_array[idx, :] = image.flatten()
        y_array[idx] = image_row.gender
        mask_path = image_path.parent.parent / MASKS_FOLDER
        mask_path = mask_path / (image_path.name.split('_')[0] + '_mano.bmp')
        mask: np.array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        m_array[idx, :] = mask.flatten()
        l_array[idx] = image_path

    return x_array, y_array, m_array, l_array
