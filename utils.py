from datetime import timedelta
from itertools import product
from time import time
from pathlib import Path
import pickle
from shutil import move

import numpy as np

from constants import datasets


class Timer:
    def __init__(self, msg: str):
        self.msg = msg
        self.start_time = None

    def start(self):
        self.start_time = time()

    def stop(self):
        if self.start_time is None:
            raise Exception('Must set start time.')
        end = time()
        delta = timedelta(seconds=end - self.start_time)
        print(self.msg, str(delta))


def find_dataset_shape(dataset_name: str):
    """Finds the dataset shape from its name. Output format is
    (rows, cols).
    """
    shape = dataset_name.split('_')[1]
    shape = tuple(int(i) for i in shape.split('x')[::-1])
    return shape


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
    # Move files
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
                    ha='center', va='center', color='w')
    fig.tight_layout()
    return fig, ax


def generate_cv_grid_plot(dataset_name: str, cv: int, params_folder: str):
    """Generates visualizations of the cross-validation results for the
    chosen dataset.
    """
    assert cv in (1, 2), 'cv must be 1 or 2'
    cv_file = Path(params_folder) / f'cv{cv}_{dataset_name}.pickle'
    if not cv_file.exists():
        raise FileNotFoundError(f'{params_folder}/{cv_file.name} not found')
    with open(cv_file, 'rb') as f:
        cv_results: dict = pickle.load(f)
    params = cv_results['params']
    results = cv_results['mean_test_score']
    results_std = cv_results['std_test_score']
    # Get unique values of params
    param_types = list(params[0].keys())
    name_a = param_types[0]
    name_b = param_types[1]
    list_a = np.unique(np.array([p[name_a] for p in params]))
    list_b = np.unique(np.array([p[name_b] for p in params]))
    # Convert results to a grid
    n_a = len(list_a)
    n_b = len(list_b)
    results_grid = np.zeros((n_a, n_b), dtype='float64')
    std_grid = np.zeros((n_a, n_b), dtype='float64')
    for a in range(n_a):
        for b in range(n_b):
            i = b + n_b*a
            results_grid[a, b] = results[i]
            std_grid[a, b] = results_std[i]
    # Generate results plot
    fig, ax = grid_plot(list_a, list_b, results_grid * 100)
    ax.set_ylabel(name_a)
    ax.set_xlabel(name_b)
    ax.set_title(f'{dataset_name}, CV{cv} results')
    # Generate results std plot
    fig_std, ax_std = grid_plot(list_a, list_b, std_grid * 100)
    ax_std.set_ylabel(name_a)
    ax_std.set_xlabel(name_b)
    ax_std.set_title(f'{dataset_name}, CV{cv} results, std')
    return fig, fig_std


def review_cv_results(params_folder: str, out_folder: str):
    import matplotlib.pyplot as plt
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        for cv in (1, 2):
            try:
                fig, fig_std = generate_cv_grid_plot(dataset, cv, params_folder)
            except FileNotFoundError:
                print(f'File for {dataset} CV{cv} not found. Skipping.')
                continue
            figname = f'{dataset}_cv{cv}.png'
            figstdname = f'{dataset}_cv{cv}_std.png'
            fig.savefig(str(out_folder / figname), bbox_inches='tight',
                        transparent=False)
            fig_std.savefig(str(out_folder / figstdname), bbox_inches='tight',
                            transparent=False)
            plt.close('all')


def review_results(results_folder: str):
    results_folder = Path(results_folder)
    for file in results_folder.glob('*.pickle'):
        with open(file, 'rb') as f:
            cur_results = pickle.load(f)
        cur_results = np.array([d['accuracy'] for d in cur_results])
        mean = cur_results.mean() * 100
        std = cur_results.std() * 100
        print(f'{file.stem}:\t{mean:.2f} ± {std:.2f}')


def generate_mask_visualization(dataset_name: str, pairs: str,
                                partition=1):
    """Generates a grayscale visualization of the masks of the dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use
    pairs : str or None/False
        Set to None/False if pairs are not to be used. Otherwise, set to the
        pairing method name.
    partition : int
        Train partition to use. Default 1.
    """
    from load_partitions import load_partitions_pairs
    _, _, train_m, _, _, _, _, _ = load_partitions_pairs(
        dataset_name, partition, 0, True, pairs)
    masks = train_m.mean(axis=0)
    masks = masks * 255
    shape = find_dataset_shape(dataset_name)
    masks = masks.reshape(shape)
    return masks.astype('uint8')
