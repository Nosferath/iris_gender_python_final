import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from constants import datasets, MASK_VALUE, SCALE_DATASET, N_PARTS, PAIR_METHOD
from generate_subindexes import generate_subindexes
from load_partitions import load_partitions, load_partitions_pairs
from pairs import load_pairs_array
from utils import Timer


def get_ada_param_grid(start_nestimators, step_nestimators, end_nestimators,
                       startlog2_lr, steplog2_lr, endlog2_lr):
    nsteps_nestimators = int(np.floor(
        (end_nestimators - start_nestimators + step_nestimators)
        / step_nestimators
    ))
    nsteps_log2lr = int(np.floor(
        (endlog2_lr - startlog2_lr + steplog2_lr) / steplog2_lr
    ))
    nestimators = np.linspace(start_nestimators, end_nestimators,
                              nsteps_nestimators, dtype=int)
    lr = np.logspace(startlog2_lr, endlog2_lr, nsteps_log2lr, base=2)
    param_grid = {'n_estimators': nestimators,
                  'learning_rate': lr}
    return param_grid


def find_best_ada_params(train_x: np.ndarray, train_y: np.ndarray,
                         dataset_name: str, partition: int, folder_name: str,
                         pair_method: str):
    # Create out folder
    out_folder = Path(folder_name)
    out_folder.mkdir(exist_ok=True)
    # Generate CV1 param. grid
    start_nestimators = 200
    step_nestimators = 200
    end_nestimators = 1000
    startlog2_lr = -5
    steplog2_lr = 1
    endlog2_lr = 1
    param_grid = get_ada_param_grid(start_nestimators, step_nestimators,
                                    end_nestimators, startlog2_lr, steplog2_lr,
                                    endlog2_lr)
    # Get subindexes
    pairs = load_pairs_array(dataset_name=dataset_name,
                             pair_method=pair_method,
                             partition=partition)
    subindexes = generate_subindexes(pairs)
    # First CV
    if dataset_name != 'left_240x20_fixed':  # TODO DELETE
        t = Timer(f'Ada CV1 {dataset_name} execution time:')
        ada = GridSearchCV(AdaBoostClassifier(), param_grid, n_jobs=-1,
                           cv=PredefinedSplit(subindexes), verbose=1)
        t.start()
        ada.fit(train_x, train_y)
        t.stop()
        # Store CV1 results
        cv1_results = ada.cv_results_
        with open(out_folder / ('cv1_' + dataset_name + '.pickle'), 'wb') as f:
            pickle.dump(cv1_results, f)
        best_nestimators = ada.best_params_['n_estimators']
        best_log2lr = np.log2(ada.best_params_['learning_rate'])
    else:
        with open(out_folder / ('cv1_' + dataset_name + '.pickle'), 'rb') as f:
            cv1_results = pickle.load(f)
        best_nestimators = cv1_results['n_estimators']
        best_log2lr = np.log2(cv1_results['learning_rate'])
    # Generate CV2 param. grid
    start_nestimators = max(best_nestimators - step_nestimators, 1)
    end_nestimators = best_nestimators + step_nestimators
    step_nestimators = step_nestimators / 5
    startlog2_lr = best_log2lr - steplog2_lr
    endlog2_lr = best_log2lr + steplog2_lr
    steplog2_lr = steplog2_lr / 5
    param_grid = get_ada_param_grid(start_nestimators, step_nestimators,
                                    end_nestimators, startlog2_lr, steplog2_lr,
                                    endlog2_lr)
    # Second CV
    t = Timer(f'ADA CV2 {dataset_name} execution time:')
    ada = GridSearchCV(AdaBoostClassifier(), param_grid, n_jobs=-1,
                       cv=PredefinedSplit(subindexes), verbose=1)
    t.start()
    ada.fit(train_x, train_y)
    t.stop()
    # Store CV2 results
    cv2_results = ada.cv_results_
    with open(out_folder / ('cv2' + dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_results, f)
    cv2_params = ada.best_params_
    with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_params, f)

    return cv2_params


def main(find_params=True):
    out_params_name = 'ada_params'
    # Find best ADA params
    params_partition = 1
    params_list = []
    if find_params:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            train_x, train_y, _, _, _, _, _, _, = load_partitions(
                dataset_name, params_partition, MASK_VALUE, SCALE_DATASET
            )
            params = find_best_ada_params(train_x,
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

    # Perform ADA test
    out_folder = Path('ada_results')
    out_folder.mkdir(exist_ok=True)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        params: dict = params_list[data_idx]
        nestimators = params['n_estimators']
        lr = params['learning_rate']
        results = []
        for part in range(1, N_PARTS):
            train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
                dataset_name, part, MASK_VALUE, SCALE_DATASET
            )
            ada = AdaBoostClassifier(nestimators, learning_rate=lr,
                                     n_jobs=-1, random_state=42)
            ada.fit(train_x, train_y)
            predicted = ada.predict(test_x)
            cur_results = classification_report(test_y, predicted,
                                                output_dict=True)
            results.append(cur_results)
            with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
                pickle.dump(results, f)


def main_std(find_params=True):
    out_params_name = 'ada_params_std'
    # Find best ADA params
    params_partition = 1
    params_list = []
    if find_params:
        for data_idx in range(len(datasets)):
            dataset_name = datasets[data_idx]
            train_x, train_y, _, _, _, _, _, _, = load_partitions_pairs(
                dataset_name, params_partition, MASK_VALUE, SCALE_DATASET,
                PAIR_METHOD
            )
            params = find_best_ada_params(train_x,
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

    # Perform ADA test
    out_folder = Path('ada_results_std')
    out_folder.mkdir(exist_ok=True)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        params: dict = params_list[data_idx]
        nestimators = params['n_estimators']
        lr = params['learning_rate']
        results = []
        for part in range(1, N_PARTS):
            train_x, train_y, _, _, test_x, test_y, _, _ = \
                load_partitions_pairs(
                    dataset_name, part, MASK_VALUE, SCALE_DATASET, PAIR_METHOD
                )
            ada = AdaBoostClassifier(nestimators, learning_rate=lr,
                                     n_jobs=-1, random_state=42)
            ada.fit(train_x, train_y)
            predicted = ada.predict(test_x)
            cur_results = classification_report(test_y, predicted,
                                                output_dict=True)
            results.append(cur_results)
            with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
                pickle.dump(results, f)
