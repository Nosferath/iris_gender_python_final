from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from load_partitions import load_partitions, load_partitions_pairs, \
    load_partitions_pairs_excl
from pairs import prepare_pairs_indexes, load_pairs_array
from generate_subindexes import generate_subindexes
from utils import Timer


MASK_VALUE = 0
TEST_PROPORTION = 0.4
SCALE_DATASET = True
N_PARTS = 10
PAIR_METHOD = 'agrowth_hung_10'


def get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees, start_maxfeats,
                      step_maxfeats, end_maxfeats):
    nsteps_ntrees = int(np.floor((end_ntrees - start_ntrees + step_ntrees)
                                 / step_ntrees))
    nsteps_maxfeats = int(np.floor((end_maxfeats - start_maxfeats + step_maxfeats)
                                   / step_maxfeats))
    ntrees = np.linspace(start_ntrees, end_ntrees, nsteps_ntrees, dtype=int)
    maxfeats = np.linspace(start_maxfeats, end_maxfeats, nsteps_maxfeats,
                           dtype=int)
    param_grid = {'n_estimators': np.unique(ntrees),
                  'max_features': np.unique(maxfeats)}
    return param_grid


def find_best_rf_params(train_x: np.ndarray, train_y: np.ndarray,
                        dataset_name: str, partition: int, folder_name: str,
                        pair_method: str):
    # Create out folder
    out_folder = Path(folder_name)
    out_folder.mkdir(exist_ok=True)
    # Generate CV1 param. grid
    start_ntrees = 200
    step_ntrees = 200
    end_ntrees = 2000
    sqrt_feats = np.round(np.sqrt(train_x.shape[1]))
    start_maxfeats = int(np.floor(sqrt_feats / 2.0))
    step_maxfeats = int(np.ceil(start_maxfeats / 2.0))
    end_maxfeats = start_maxfeats + 4*step_maxfeats
    param_grid = get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees,
                                   start_maxfeats, step_maxfeats, end_maxfeats)
    # Get subindexes
    pairs = load_pairs_array(dataset_name=dataset_name,
                             pair_method=pair_method,
                             partition=partition)
    subindexes = generate_subindexes(pairs)
    # First CV
    t = Timer(f'RF CV1 {dataset_name} execution time:')
    rf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                      cv=PredefinedSplit(subindexes), verbose=1)
    t.start()  # DEBUG
    rf.fit(train_x, train_y)
    t.stop()  # DEBUG
    # Store CV1 results
    cv1_results = rf.cv_results_
    with open(out_folder / ('cv1_' + dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv1_results, f)
    # Generate CV2 param. grid
    best_ntrees = rf.best_params_['n_estimators']
    best_maxfeats = rf.best_params_['max_features']
    start_ntrees = max(best_ntrees - step_ntrees, 1)
    end_ntrees = best_ntrees + step_ntrees
    step_ntrees = int(step_ntrees / 5)
    start_maxfeats = best_maxfeats - step_maxfeats
    end_maxfeats = best_maxfeats + step_maxfeats
    step_maxfeats = int(np.ceil(step_maxfeats / 5))
    param_grid = get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees,
                                   start_maxfeats, step_maxfeats, end_maxfeats)
    # Second CV
    t = Timer(f'RF CV2 {dataset_name} execution time:')
    rf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                      cv=PredefinedSplit(subindexes), verbose=1)
    t.start()  # DEBUG
    rf.fit(train_x, train_y)
    t.stop()  # DEBUG
    # Store CV2 results
    cv2_results = rf.cv_results_
    with open(out_folder / ('cv2_' + dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_results, f)
    cv2_params = rf.best_params_
    with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_params, f)

    return cv2_params


def main(find_params=True):
    datasets = ('left_240x20_fixed', 'right_240x20_fixed', 'left_240x40_fixed',
                'right_240x40_fixed', 'left_480x80_fixed',
                'right_480x80_fixed', 'left_240x20', 'right_240x20',
                'left_240x40', 'right_240x40', 'left_480x80', 'right_480x80')
    out_params_name = 'rf_params'
    # Find best RF params
    params_partition = 1
    params_list = []
    if find_params:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            train_x, train_y, _, _, _, _, _, _, = load_partitions(
                dataset_name, params_partition, MASK_VALUE, SCALE_DATASET)
            params = find_best_rf_params(train_x,
                                         train_y,
                                         dataset_name,
                                         params_partition,
                                         out_params_name,
                                         PAIR_METHOD)
            params_list.append(params)
    else:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            with open(Path.cwd() / out_params_name
                      / (dataset_name + '.pickle'), 'rb') as f:
                params = pickle.load(f)
                params_list.append(params)

    # Perform RF test
    out_folder = Path('rf_results')
    out_folder.mkdir(exist_ok=True)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        params: dict = params_list[data_idx]
        ntrees = params['n_estimators']
        max_feats = params['max_features']
        results = []
        for part in range(1, N_PARTS + 1):
            train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
                dataset_name, part, MASK_VALUE, SCALE_DATASET
            )
            rf = RandomForestClassifier(ntrees, max_features=max_feats,
                                        n_jobs=-1, random_state=42)
            rf.fit(train_x, train_y)
            predicted = rf.predict(test_x)
            cur_results = classification_report(test_y, predicted,
                                                output_dict=True)
            results.append(cur_results)
            with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
                pickle.dump(results, f)


def main_std(find_params=True):
    datasets = ('left_240x20_fixed', 'right_240x20_fixed', 'left_240x40_fixed',
                'right_240x40_fixed', 'left_480x80_fixed',
                'right_480x80_fixed', 'left_240x20', 'right_240x20',
                'left_240x40', 'right_240x40', 'left_480x80', 'right_480x80')
    out_params_name = 'rf_params_std'
    # Find best RF params
    params_partition = 1
    params_list = []
    if find_params:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            train_x, train_y, _, _, _, _, _, _, = load_partitions_pairs(
                dataset_name, params_partition, MASK_VALUE, SCALE_DATASET,
                PAIR_METHOD
            )
            params = find_best_rf_params(train_x,
                                         train_y,
                                         dataset_name,
                                         params_partition,
                                         out_params_name,
                                         PAIR_METHOD)
            params_list.append(params)
    else:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            with open(Path.cwd() / out_params_name
                      / (dataset_name + '.pickle'), 'rb') as f:
                params = pickle.load(f)
                params_list.append(params)

    # Perform RF test
    out_folder = Path('rf_results_std')
    out_folder.mkdir(exist_ok=True)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        params: dict = params_list[data_idx]
        ntrees = params['n_estimators']
        max_feats = params['max_features']
        results = []
        for part in range(1, N_PARTS + 1):
            train_x, train_y, _, _, test_x, test_y, _, _ = \
                load_partitions_pairs(
                    dataset_name, part, MASK_VALUE, SCALE_DATASET, PAIR_METHOD
            )
            rf = RandomForestClassifier(ntrees, max_features=max_feats,
                                        n_jobs=-1, random_state=42)
            rf.fit(train_x, train_y)
            predicted = rf.predict(test_x)
            cur_results = classification_report(test_y, predicted,
                                                output_dict=True)
            results.append(cur_results)
            with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    main()
