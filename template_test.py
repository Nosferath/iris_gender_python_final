import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from constants import datasets, MASK_VALUE, SCALE_DATASET, PAIR_METHOD, N_PARTS, PARAMS_PARTITION
from generate_subindexes import generate_subindexes
from load_partitions import load_partitions, load_partitions_pairs
from pairs import load_pairs_array
from utils import Timer


def get_param_grid(start_param_a, step_param_a, end_param_a,
                   start_param_b, step_param_b, end_param_b,
                   param_a_name: str, param_b_name: str,  # Names for the dict
                   param_a_func: str, param_b_func: str,
                   type_a, type_b):
    """Base function for the get_x_param_grid functions."""
    nsteps_a = int(np.floor((end_param_a - start_param_a + step_param_a)
                            / step_param_a))
    nsteps_b = int(np.floor((end_param_b - start_param_b + step_param_b)
                            / step_param_b))

    def get_param(start_param, end_param, nsteps, type_param, param_func):
        if param_func == 'linspace':
            param = np.linspace(start_param, end_param, nsteps,
                                dtype=type_param)
        elif param_func == 'logspace2':
            param = np.logspace(start_param, end_param, nsteps, base=2,
                                dtype=type_param)
        else:
            raise NotImplemented('This function has not been implemented for'
                                 'params.')
        return param

    param_a = get_param(start_param_a, end_param_a, nsteps_a, type_a,
                        param_a_func)
    param_b = get_param(start_param_b, end_param_b, nsteps_b, type_b,
                        param_b_func)
    param_grid = {param_a_name: param_a,
                  param_b_name: param_b}
    return param_grid


def find_best_params(train_x: np.ndarray, train_y: np.ndarray,
                     dataset_name: str, partition: int, folder_name: str,
                     pair_method: str,
                     start_param_a, step_param_a, end_param_a,
                     start_param_b, step_param_b, end_param_b,
                     param_a_islog2: bool, param_b_islog2: bool,
                     param_a_min1: bool, param_b_min1: bool,
                     param_grid_fn, clasif_name: str, clasif_fn,
                     n_jobs: int):
    """Base function for finding classifier params.

    Parameters
    ----------
    train_x
    train_y
    dataset_name
    partition
    folder_name
    pair_method
    start_param_a
    step_param_a
    end_param_a
    start_param_b
    step_param_b
    end_param_b
    param_a_islog2, param_b_islog2 : bool
        Indicate whether the param. is indicated as a log. of 2
    param_a_min1, param_b_min1 : bool
        Indicate whether the param. has to be at least 1.
    param_grid_fn
        Parameter grid function for the classifier
    clasif_name : str
        Name of the classifier. Used for printing elapsed time.
    clasif_fn : sklearn classifier class function
    n_jobs : int
        Number of parallel workers to use. -1 uses all available.
    """
    # Create out folder
    out_folder = Path(folder_name)
    out_folder.mkdir(exist_ok=True)
    # Generate CV1 param. grid
    param_grid = param_grid_fn(start_param_a, step_param_a, end_param_a,
                               start_param_b, step_param_b, end_param_b)
    # Get subindexes
    pairs = load_pairs_array(dataset_name=dataset_name,
                             pair_method=pair_method,
                             partition=partition)
    subindexes = generate_subindexes(pairs)
    # First CV
    t = Timer(f'{clasif_name} CV1 {dataset_name} execution time:')
    model = GridSearchCV(clasif_fn(), param_grid, n_jobs=n_jobs,
                         cv=PredefinedSplit(subindexes), verbose=1)
    t.start()
    model.fit(train_x, train_y)
    t.stop()
    # Store CV1 results
    cv1_results = model.cv_results_
    with open(out_folder / ('cv1_' + dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv1_results, f)

    # Generate CV2 param. grid
    params = list(param_grid.keys())
    best_a = model.best_params_[params[0]]
    if param_a_islog2:
        best_a = np.log2(best_a)
    best_b = model.best_params_[params[1]]
    if param_b_islog2:
        best_b = np.log2(best_b)
    start_param_a = best_a - step_param_a
    if param_a_min1:
        start_param_a = max(start_param_a, 1)
    end_param_a = best_a + step_param_a
    step_param_a /= 2
    start_param_b = best_b - step_param_b
    if param_b_min1:
        start_param_b = max(start_param_b, 1)
    end_param_b = best_b + step_param_b
    step_param_b /= 2
    param_grid = param_grid_fn(start_param_a, step_param_a, end_param_a,
                               start_param_b, step_param_b, end_param_b)
    # Second CV
    t = Timer(f'{clasif_name} CV2 {dataset_name} execution time:')
    model = GridSearchCV(clasif_fn(), param_grid, n_jobs=n_jobs,
                         cv=PredefinedSplit(subindexes), verbose=1)
    t.start()
    model.fit(train_x, train_y)
    t.stop()
    # Store CV2 results
    cv2_results = model.cv_results_
    with open(out_folder / ('cv2_' + dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_results, f)
    cv2_params = model.best_params_
    with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
        pickle.dump(cv2_params, f)

    return cv2_params


def main_base(find_params: bool, out_params_name: str, find_params_fn,
              out_results_name: str, clasif_fn, use_std_masks: bool,
              n_cmim: int, n_jobs: int):
    """Base function for running classifier tests.

    Parameters
    ----------
    find_params : bool
        If True, parameters for the model will be obtained using the
        find_params_fn, and any pre-existing parameters will be
        overwritten. If False, parameters will be loaded from file.
    out_params_name : str
        Name of the folder where the parameters will be stored to or
        loaded from.
    find_params_fn
        Function that will be used for finding the parameters.
    out_results_name : str
        Name of the folder where the results will be stored.
    clasif_fn
        Classifier function. Normally, a function from sklearn.
    use_std_masks : bool
        If true, mask pairs will be loaded from pairs/<folder>, where
        folder is the one set in constants.PAIR_METHOD, and will be
        applied to the dataset.
    n_cmim : int
        Must be 0 or greater. If 0, CMIM will not be used. Otherwise, it
        indicates how many feature groups will be used. How many groups
        are there is set in constants.CMIM_GROUPS.
    n_jobs : int
        Sets the number of jobs for parallel tasks. If -1, it will use
        all workers available. Recommended to set as max - 1 so the
        computer does not get stuck it is going to be used while it
        iterates.
    """
    # Find best model params
    params_list = []
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        if find_params:
            if use_std_masks:
                train_x, train_y, _, _, _, _, _, _, = load_partitions_pairs(
                    dataset_name, PARAMS_PARTITION, MASK_VALUE, SCALE_DATASET,
                    PAIR_METHOD
                )
                # TODO add CMIM
            else:
                train_x, train_y, _, _, _, _, _, _, = load_partitions(
                    dataset_name, PARAMS_PARTITION, MASK_VALUE, SCALE_DATASET
                )
                # TODO add CMIM
            params = find_params_fn(train_x=train_x,
                                    train_y=train_y,
                                    dataset_name=dataset_name,
                                    partition=PARAMS_PARTITION,
                                    folder_name=out_params_name,
                                    pair_method=PAIR_METHOD,
                                    n_jobs=n_jobs)
        else:
            with open(Path.cwd() / out_params_name
                      / (dataset_name + '.pickle'), 'rb') as f:
                params = pickle.load(f)
        params_list.append(params)

    # Perform model test
    out_folder = Path(out_results_name)
    out_folder.mkdir(exist_ok=True, parents=True)
    for data_idx in range(len(datasets)):
        dataset_name = datasets[data_idx]
        params: dict = params_list[data_idx]
        results = []
        for part in range(1, N_PARTS + 1):
            if use_std_masks:
                train_x, train_y, _, _, test_x, test_y, _, _ = \
                    load_partitions_pairs(
                        dataset_name, part, MASK_VALUE, SCALE_DATASET, PAIR_METHOD
                    )
            else:
                train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
                    dataset_name, part, MASK_VALUE, SCALE_DATASET
                )
            try:
                model = clasif_fn(**params, n_jobs=n_jobs, random_state=42)
            except TypeError:
                model = clasif_fn(**params, random_state=42)
            model.fit(train_x, train_y)
            predicted = model.predict(test_x)
            cur_results = classification_report(test_y, predicted,
                                                output_dict=True)
            results.append(cur_results)
            with open(out_folder / (dataset_name + '.pickle'), 'wb') as f:
                pickle.dump(results, f)
