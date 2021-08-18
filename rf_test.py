from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from load_partitions import load_partitions, load_partitions_pairs, \
    load_partitions_pairs_excl, prepare_pairs_indexes


MASK_VALUE = 0
TEST_PROPORTION = 0.4
SCALE_DATASET = True
N_PARTS = 10
PAIR_METHOD = 'agrowth_hung_10'


def generate_subindexes(pairs: np.ndarray):
    """Generate partition subindexes that are based on pairs, so that
    elements of the same pair are always in the same partition.
    (For Step 9)

    The returned array contains the fold index for each element.
    """
    pairs = prepare_pairs_indexes(pairs)
    n_pairs = pairs.shape[0]
    kfold = KFold(n_splits=5, shuffle=False, random_state=42)
    subindexes = np.full(2 * n_pairs, -1, dtype=int)
    for fold_idx, fold in enumerate(kfold.split(np.arange(n_pairs))):
        _, cur_fold = fold
        for pair_idx in cur_fold:
            cur_pair = pairs[pair_idx, :]
            subindexes[cur_pair] = fold_idx
    assert -1 not in subindexes, "Not all elements were set to a fold"
    return subindexes


def find_best_rf_params(train_x: np.ndarray, train_y: np.ndarray,
                        dataset_name: str, partition: int, folder_name: str,
                        pair_method: str):
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

