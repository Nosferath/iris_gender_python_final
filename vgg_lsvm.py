import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from constants import TEST_SIZE
from load_data_utils import partition_data, partition_both_eyes
from utils import Timer
from vgg_utils import load_data


def vgg_feat_lsvm_parall(data, partition: int, n_iters: Union[int, None],
                         both_eyes_mode: bool, params_set='norm3', n_jobs=1):
    """Parallelizable function that performs the VGG-feat Linear-SVM
    test. If GridSearch is desired, n_iters should be None.

    n_jobs should be 1 unless it is not being parallelized from outside.
    """
    # Prepare data
    if both_eyes_mode:
        all_data, males_set, females_set = data
        train_x, train_y, test_x, test_y = partition_both_eyes(
            all_data, males_set, females_set, TEST_SIZE, partition
        )
    else:
        data_x, labels = data
        train_x, train_y, test_x, test_y = partition_data(
            data_x, labels, TEST_SIZE, partition
        )
    # Train model
    if n_iters is not None:
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', LinearSVC(max_iter=n_iters,
                                random_state=42))
        ])
    else:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', LinearSVC(random_state=42))
        ])
        if params_set == 'norm3':
            param_grid = {'model__max_iter': [50, 100, 300, 500, 1000],
                          # np.linspace(1000, 5000, 3),
                          # 'model__C': np.linspace(0.125, 1.0, 8)}
                          'model__C': np.logspace(-7, -3, 5, base=2)}
        elif params_set == 'norm4':
            param_grid = {'model__max_iter': [25, 50, 100, 300],
                          # np.linspace(1000, 5000, 3),
                          # 'model__C': np.linspace(0.125, 1.0, 8)}
                          'model__C': np.logspace(-11, -7, 5, base=2)}
        elif params_set == 'peri3':
            param_grid = {'model__max_iter': np.linspace(1000, 6000, 6),
                          'model__C': np.linspace(0.5, 2.0, 4)}
        else:
            raise ValueError('params_set option not recognized')

        model = GridSearchCV(pipe, param_grid, cv=5, n_jobs=n_jobs,
                             random_state=42)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    results = classification_report(test_y, pred, output_dict=True)
    results['cv_results'] = model.cv_results_

    del train_x, train_y, test_x, test_y
    return results


def _perform_vgg_feat_lsvm_test(data_type, data_params, dataset_name: str,
                                n_partitions: int, n_jobs: int, out_folder,
                                n_iters: int, both_eyes_mode: bool,
                                parallel=True):
    """Performs VGG feat lsvm test."""
    # Load data
    if both_eyes_mode:
        msg = 'VGG Features, LSVM Test, both eyes dataset'
    else:
        msg = 'VGG Features, LSVM Test'
    t = Timer(f'Loading dataset {data_type}, params {data_params}')
    t.start()
    data = load_data(data_type, **data_params)
    t.stop()
    if 'periocular' in data_type:
        params_set = 'peri3'
    else:
        params_set = 'norm4'

    # Perform parallel test
    subjobs = 1 if parallel else n_jobs
    args = [(data, i, n_iters, both_eyes_mode, params_set, subjobs)
            for i in range(n_partitions)]
    print(msg)
    if parallel:
        with Pool(n_jobs) as p:
            t = Timer(f"{dataset_name}, {n_partitions} "
                      f"partitions, {n_jobs} jobs")
            t.start()
            results = p.starmap(vgg_feat_lsvm_parall, args)
            t.stop()
    else:
        t = Timer(f"{dataset_name}, {n_partitions} partitions, {n_jobs} jobs")
        t.start()
        results = []
        for arg in args:
            result = vgg_feat_lsvm_parall(*arg)
            results.append(result)
        t.stop()

    # Store results
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}.pickle', 'wb') as f:
        pickle.dump(results, f)
    del args, data


def perform_vgg_feat_lsvm_test(dataset_name: str, n_partitions: int,
                               n_jobs: int,
                               out_folder='vgg_feat_lsvm_results',
                               n_iters: int = 10000,
                               parallel=True):
    data_type = 'iris_vgg_feats'
    data_params = {'dataset_name': dataset_name}
    _perform_vgg_feat_lsvm_test(data_type, data_params, dataset_name,
                                n_partitions, n_jobs, out_folder, n_iters,
                                both_eyes_mode=False, parallel=parallel)


def perform_peri_vgg_feat_lsvm_test(eye: str, n_partitions: int, n_jobs: int,
                                    out_folder='vgg_feat_lsvm_peri_results',
                                    n_iters: int = 10000, parallel=True):
    data_type = 'periocular_vgg_feats'
    data_params = {'eye': eye}
    _perform_vgg_feat_lsvm_test(data_type, data_params, eye, n_partitions,
                                n_jobs, out_folder, n_iters,
                                both_eyes_mode=False, parallel=parallel)


def perform_vgg_feat_lsvm_test_botheyes(
        dataset_name: str, n_partitions: int, n_jobs: int,
        out_folder='vgg_feat_lsvm_botheyes_results', n_iters: int = 10000,
        parallel=True
):
    data_type = 'iris_botheyes_vgg_feats'
    data_params = {'dataset_name': dataset_name}
    _perform_vgg_feat_lsvm_test(data_type, data_params, dataset_name,
                                n_partitions, n_jobs, out_folder, n_iters,
                                both_eyes_mode=True, parallel=parallel)


def perform_vgg_feat_lsvm_test_botheyes_peri(
        n_partitions: int, n_jobs: int,
        out_folder='vgg_feat_lsvm_botheyes_peri_results',
        n_iters: int = 10000, parallel=True
):
    dataset_name = 'both_peri'
    data_type = 'periocular_botheyes_vgg_feats'
    data_params = {}
    _perform_vgg_feat_lsvm_test(data_type, data_params, dataset_name,
                                n_partitions, n_jobs, out_folder, n_iters,
                                both_eyes_mode=True, parallel=parallel)


def main_vgg_feat_lsvm_test():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('n_jobs', type=int,
                    help='Number of jobs')
    ap.add_argument('-p', '--n_parts', type=int, default=10,
                    help='Number of random partitions to test on')
    ap.add_argument('-i', '--n_iters', type=int, default=None,
                    help='Number of iterations for LSVM')
    ap.add_argument('--use_botheyes', action='store_true',
                    help='Perform test using both eyes')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    ap.add_argument('--out_folder', type=str, default=None,
                    help='Out folder for the test')
    ap.add_argument('--no_parallel', action='store_true',
                    help='Do not parallelize from outside, use GridSearchCV '
                         'n_jobs instead.')
    args = ap.parse_args()
    n_jobs = args.n_jobs
    n_parts = args.n_parts
    n_iters = args.n_iters
    use_botheyes = args.use_botheyes
    use_peri = args.use_peri
    out_folder = args.out_folder
    no_parallel = args.no_parallel
    if n_iters is None:
        folder_suffix = ''
    else:
        folder_suffix = f'_{n_iters / 1000:0.0f}k_iters'

    if not use_botheyes:
        from constants import datasets
        if not use_peri:
            if out_folder is None:
                out_folder = f'vgg_feat_lsvm_results{folder_suffix}'
            for d in datasets:
                perform_vgg_feat_lsvm_test(d, n_parts, n_jobs,
                                           out_folder=out_folder,
                                           n_iters=n_iters,
                                           parallel=not no_parallel)
        else:
            if out_folder is None:
                out_folder = f'vgg_feat_lsvm_peri_results{folder_suffix}'
            for eye in ('left', 'right',):
                perform_peri_vgg_feat_lsvm_test(eye, n_parts, n_jobs,
                                                out_folder=out_folder,
                                                n_iters=n_iters,
                                                parallel=not no_parallel)

    else:
        from constants import datasets_botheyes
        if not use_peri:
            if out_folder is None:
                out_folder = f'vgg_feat_lsvm_botheyes_results{folder_suffix}'
            for d in datasets_botheyes:
                perform_vgg_feat_lsvm_test_botheyes(d, n_parts, n_jobs,
                                                    out_folder=out_folder,
                                                    n_iters=n_iters,
                                                    parallel=not no_parallel)
        else:
            if out_folder is None:
                out_folder = f'vgg_feat_lsvm_botheyes_peri_results' \
                             f'{folder_suffix}'
            perform_vgg_feat_lsvm_test_botheyes_peri(n_parts, n_jobs,
                                                     out_folder=out_folder,
                                                     n_iters=n_iters,
                                                     parallel=not no_parallel)


if __name__ == '__main__':
    main_vgg_feat_lsvm_test()
