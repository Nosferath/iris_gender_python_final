from datetime import timedelta
from itertools import product
from time import time
from pathlib import Path
from shutil import move

import pandas as pd


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


def process_ndiris(root_folder, csv_path, dest_folder):
    """Separates ND-Iris dataset folders into subfolders according
    to sensor and eye, and gathers the labels of every image."""
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
