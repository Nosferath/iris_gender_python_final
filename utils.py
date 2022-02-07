from datetime import timedelta
from itertools import product
from time import perf_counter
from pathlib import Path
from shutil import move

import numpy as np


class Timer:
    def __init__(self, msg: str):
        self.msg = msg
        self.start_time = None

    def start(self):
        print('Starting timer,', self.msg)
        self.start_time = perf_counter()

    def stop(self):
        if self.start_time is None:
            raise Exception('Must set start time.')
        end = perf_counter()
        delta = timedelta(seconds=end - self.start_time)
        print(self.msg, str(delta))


def find_dataset_shape(dataset_name: str):
    """Finds the dataset shape from its name. Output format is
    (rows, cols).
    """
    shape = dataset_name.split('_')[1]
    shape = tuple(int(i) for i in shape.split('x')[::-1])
    return shape


def find_n_features(datase_name: str):
    """Finds the number of features from datatset name."""
    shape = find_dataset_shape(datase_name)
    return shape[0]*shape[1]


def find_shape(n_features: int = None, dataset_name: str = None):
    if n_features is None and dataset_name is None \
            or n_features is not None and dataset_name is not None:
        raise ValueError(
            'You must set EITHER n_features or dataset_name'
        )
    if n_features is None:
        dataset_name: str
        return find_dataset_shape(dataset_name)

    shapes = {4800: (20, 240), 9600: (40, 240), 38400: (80, 480)}
    return shapes[n_features]


def process_ndiris(root_folder, csv_path, dest_folder):
    """Separates ND-Iris dataset folders into subfolders according
    to sensor and eye, and gathers the labels of every image."""
    import pandas as pd
    # Initialize destination folders
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(exist_ok=True)
    genders = {'Male': 0, 'Female': 1}
    sensors = ('LG4000', 'LG2200')
    eyes = ('left', 'right')
    folders = ('NormalizedImages', 'NormalizedMasks', 'SegmentedImages')
    for s in sensors:
        sensor_folder = dest_folder / s
        sensor_folder.mkdir(exist_ok=True)
        for e in eyes:
            eye_folder = sensor_folder / e
            eye_folder.mkdir(exist_ok=True)
            for f in folders:
                cur_folder = eye_folder / f
                cur_folder.mkdir(exist_ok=True)
    gender_data = {(s, e): pd.DataFrame(columns=['img_id', 'gender'])
                   for s, e in product(sensors, eyes)}
    # Load and clean dataframe
    df = pd.read_csv(csv_path)
    df = df[df['sensor'] != 'resize']
    # df['eye'] = df['eye'].apply(lambda x: str(x).lower())
    # Move images
    root_folder = Path(root_folder)
    assert root_folder.exists()
    for f in folders:
        img_list = list((root_folder / f).glob('*.bmp'))
        for p in img_list:
            img_id = p.stem.split('_')[0]
            filename = img_id + '.tiff'
            cur_row = df[df['filename'] == filename]
            if len(cur_row) != 1:
                print('Failed to find', filename, 'row len was',
                      str(len(cur_row)))
                continue
            cur_row = cur_row.squeeze()
            gender = genders[cur_row['gender']]
            sensor = cur_row['sensor']
            eye = cur_row['eye'].lower()
            move(p, dest_folder / sensor / eye / f / p.name)
            gender_data[(sensor, eye)] = gender_data[(sensor, eye)].append({
                'img_id': img_id,
                'gender': gender
            }, ignore_index=True)
    for s, e in product(sensors, eyes):
        gender_data[(s, e)].to_csv(dest_folder / s / e / 'labels.csv')


def fix_folders(root_folder):
    sensors = ('LG4000', 'LG2200')
    eyes = ('left', 'right')
    folders = ('NormalizedImages', 'NormalizedMasks', 'SegmentedImages')
    root_folder = Path(root_folder)
    for s, e, f in product(sensors, eyes, folders):
        cur_folder = root_folder / s / e / f
        img_list = list(cur_folder.glob('*.bmp'))
        for p in img_list:
            move(p, root_folder / f / p.name)


def grid_plot(a: np.ndarray, b: np.ndarray, z: np.ndarray):
    """Generates a grid plot from the inputs."""
    import matplotlib.pyplot as plt
    assert len(a) == z.shape[0]
    assert len(b) == z.shape[1]
    fig, ax = plt.subplots()
    ax.imshow(z)
    # Show all ticks
    ax.set_xticks(np.arange(len(b)))
    ax.set_yticks(np.arange(len(a)))
    # Label ticks with valaues
    ax.set_xticklabels(np.round(b, 4))
    ax.set_yticklabels(np.round(a, 4))
    for i in range(len(a)):
        for j in range(len(b)):
            ax.text(j, i, np.round(z[i, j], 2),
                    ha='center', va='center', color='black')
    fig.tight_layout()
    return fig, ax


def import_matlab_results(in_root: str, in_folders: list, out_folders: list):
    import pickle
    from scipy.io import loadmat
    in_root = Path(in_root)
    for i in range(len(in_folders)):
        cur_folder = in_root / in_folders[i]
        out_folder = Path(out_folders[i])
        out_folder.mkdir(exist_ok=True, parents=True)
        cur_files = list(cur_folder.glob('*.mat'))
        for file in cur_files:
            name = file.stem
            mat = loadmat(str(file))
            results = [{'accuracy': v[0]} for v in mat['results']]
            with open(out_folder / f'{name}.pickle', 'wb') as f:
                pickle.dump(results, f)


def plot_feature_importances(importances: np.ndarray, out_path):
    import matplotlib.pyplot as plt
    cur_shape = find_shape(n_features=importances.size)
    importances = importances.reshape(cur_shape)
    plt.imshow(importances, cmap='jet')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_dmatrix(x_arr: np.ndarray, y_arr: np.ndarray):
    """Generates a DMatrix for using with XGBoost."""
    from xgboost import DMatrix
    return DMatrix(data=x_arr, label=y_arr)


def move_mod_v2():
    """This function moves all images from dataMod and the modV2 conti-
    nuous CMIM arrays to the same folders used by the regular data,
    while adding a _mod_v2 suffix to the images. This allows these images
    to be used with the same functions as the regular data.

    Used only once.
    """
    from shutil import move
    mod_folders = (Path('dataMod'), Path('cmimArraysModV2Cont'),
                   Path('cmimArraysStdModV2Cont'))
    out_folders = (Path('data'), Path('cmimArraysCont'),
                   Path('cmimArraysStdCont'))
    suf = '_mod_v2.mat'
    # Move data and CMIM images
    for cur_folder, cur_out in zip(mod_folders, out_folders):
        for file in cur_folder.glob('*.mat'):
            out_name = file.stem + suf
            move(file, cur_out / out_name)
        cur_folder.unlink()
