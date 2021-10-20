from pathlib import Path

import numpy as np

from constants import datasets, PAIR_METHOD
from utils import find_dataset_shape


def check_square_locations(out_folder='cmim_mod_checks'):
    """Generate visualizations of iris and print the index of the
     modified areas.
    """
    from PIL import Image
    from load_partitions import load_partitions_cmim, \
        load_partitions_cmim_mod_v2
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        shape = find_dataset_shape(dataset)
        for part in range(1, 3):
            for pair_method in (False, PAIR_METHOD):
                # Check that scaling is being performed correctly
                train_x, _, _, _, test_x, _, _, _ = \
                    load_partitions_cmim(
                        dataset, part, 0, True, pair_method, 0)
                x_arr = np.vstack([train_x, test_x])
                max_vals = np.max(x_arr, axis=1)
                assert np.all(max_vals == 1), 'Not all max values are 1'
                # Visualize modification and print coords
                train_x, train_y, _, train_l, test_x, test_y, _, test_l = \
                    load_partitions_cmim_mod_v2(dataset, part, 0, True,
                                             pair_method, 0)
                for i in np.random.randint(train_x.shape[0], size=4):
                    cur_x = train_x[i, :].reshape(shape)
                    cur_y = train_y[i]
                    out_name = f'{train_l[i].split(".")[0]}_{cur_y}.png'
                    out_name = dataset + '_' + str(part) + '_' + out_name
                    out_file = out_folder / out_name
                    img = Image.fromarray((cur_x*255).astype('uint8'))
                    img.save(out_file)
                for i in np.random.randint(test_x.shape[0], size=4):
                    cur_x = test_x[i, :].reshape(shape)
                    cur_y = test_y[i]
                    out_name = f'{test_l[i].split(".")[0]}_{cur_y}.png'
                    out_name = dataset + '_' + str(part) + '_' + out_name
                    out_file = out_folder / out_name
                    img = Image.fromarray((cur_x*255).astype('uint8'))
                    img.save(out_file)
