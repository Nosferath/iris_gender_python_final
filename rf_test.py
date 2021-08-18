from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from load_partitions import load_partitions, load_partitions_pairs, \
    load_partitions_pairs_excl
from pairs import prepare_pairs_indexes, load_pairs_array
from generate_subindexes import generate_subindexes


MASK_VALUE = 0
TEST_PROPORTION = 0.4
SCALE_DATASET = True
N_PARTS = 10
PAIR_METHOD = 'agrowth_hung_10'


def get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees, start_maxfeats,
                      step_maxfeats, end_maxfeats):
    nsteps_ntrees = np.floor((end_ntrees - start_ntrees + step_ntrees)
                             / step_ntrees)
    nsteps_maxfeats = np.floor((end_maxfeats - start_maxfeats + step_maxfeats)
                               / step_maxfeats)
    ntrees = np.linspace(start_ntrees, end_ntrees, nsteps_ntrees, dtype=int)
    maxfeats = np.linspace(start_maxfeats, end_maxfeats, nsteps_maxfeats,
                           dtype=int)
    param_grid = {'n_estimators': ntrees, 'max_features': maxfeats}
    return param_grid


def find_best_rf_params(train_x: np.ndarray, train_y: np.ndarray,
                        dataset_name: str, partition: int, folder_name: str,
                        pair_method: str):
    # Create out folder
    out_folder = Path(folder_name)
    out_folder.mkdir(exist_ok=True)
    # Initialize param. grid
    start_ntrees = 200
    step_ntrees = 200
    end_ntrees = 2000
    sqrt_feats = np.round(np.sqrt(train_x.shape[1]))
    start_maxfeats = np.floor(sqrt_feats / 2.0)
    step_maxfeats = np.ceil(start_maxfeats / 2.0)
    end_maxfeats = start_maxfeats + 4*step_maxfeats
    param_grid = get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees,
                                   start_maxfeats, step_maxfeats, end_maxfeats)
    # Get subindexes
    pairs = load_pairs_array(dataset_name=dataset_name,
                             pair_method=pair_method,
                             partition=partition)

    return 0


def main():
    datasets = ('left_240x20_fixed', 'right_240x20_fixed', 'left_240x40_fixed',
                'right_240x40_fixed', 'left_480x80_fixed',
                'right_480x80_fixed', 'left_240x20', 'right_240x20',
                'left_240x40', 'right_240x40', 'left_480x80', 'right_480x80')
    # Find best RF params
    params_partition = 1
    params_list = [None] * len(datasets)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        train_x, train_y, _, _, _, _, _, _, = load_partitions(
            dataset_name, params_partition, MASK_VALUE, SCALE_DATASET)
        params = find_best_rf_params(train_x,
                                     train_y,
                                     dataset_name,
                                     params_partition,
                                     'rf_params',
                                     PAIR_METHOD)
        params_list[data_idx] = params

    # Perform RF test
    out_folder = Path('rf_results')
    out_folder.mkdir(exist_ok=True)

