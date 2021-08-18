from pathlib import Path
from sys import exit

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from load_partitions import load_raw_dataset


IMAGE_SHAPE = (80, 480, 1)
# IMAGE_SHAPE = (20, 240, 1)


def reshape_and_apply_mask(data: np.ndarray, mask: np.ndarray):
    """Reshape the data vector to its final shape, and apply the mask in
    green"""
    data = data.reshape(IMAGE_SHAPE)
    data = np.tile(data, [1, 1, 3])
    mask = mask.reshape(IMAGE_SHAPE[:2])
    # Green
    masked = data[:, :, 1]
    masked[mask == 1] = 255
    data[:, :, 1] = masked
    # Others
    masked = data[:, :, 0]
    masked[mask == 1] = 0
    data[:, :, 0] = masked
    masked = data[:, :, 2]
    masked[mask == 1] = 0
    data[:, :, 2] = masked
    return data


def display_and_get_answer(image: np.ndarray, name: str):
    # Keys: 32 = Space, 13 = Enter, 27 = ESC, 49 = 1, 50 = 2
    keys = {13: 0, 32: 0, 27: None, 49: 1, 50: 2}
    cv2.imshow(name, image)
    key = cv2.waitKey()
    while key not in keys:
        key = cv2.waitKey()
    cv2.destroyAllWindows()
    return keys[key]


def main():
    # datasets = ('left_240x20_irisbee', 'right_240x20_irisbee')
    datasets = ('left_480x80', 'right_480x80')
    results_file = Path('check_masks_irisbee.csv')
    if not results_file.exists():
        results = pd.DataFrame(columns=['dataset', 'filename', 'result'])
    else:
        results = pd.read_csv(results_file, index_col=0)
    for dataset in datasets:
        data_x, data_y, data_m, data_l = load_raw_dataset(dataset)
        n_images = len(data_y)
        for i in trange(n_images, unit='img'):
            name = data_l[i, 0][0][0]
            # if name in results.filename.values:
            #     continue
            data = data_x[i, :]
            mask = data_m[i, :]
            image = reshape_and_apply_mask(data, mask)
            result = display_and_get_answer(image, name)
            if result is None:
                exit(0)
            # results = results.append({'dataset': dataset,
            #                           'filename': name,
            #                           'result': result}, ignore_index=True)
            # results.to_csv(results_file)


if __name__ == '__main__':
    main()
