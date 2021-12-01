import argparse
from pathlib import PurePath
from typing import Union

from model_tests import training_curve_test
from load_partitions import load_partitions


def perform_curve_test_on_iris(dataset_name: str, partition: int, n_jobs: int,
                               out_folder: Union[str, PurePath],
                               model_params: Union[None, dict] = None,
                               verbose: int = 1):
    from sklearn.model_selection import train_test_split

    def load_iris_tvt():
        train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
            dataset_name, partition, 0, True
        )
        test_x, val_x, test_y, val_y = train_test_split(
            test_x, test_y, test_size=0.25, stratify=test_y)
        return train_x, train_y, val_x, val_y, test_x, test_y

    model_params = {} if model_params is None else model_params
    training_curve_test(
        load_fn=load_iris_tvt,
        prepartitioned=True,
        n_jobs=n_jobs,
        out_folder=out_folder,
        model_params=model_params,
        verbose=verbose
    )


def main_iris():
    from constants import datasets
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--ntrees', type=int, required=True,
                    help="Number of estimators")
    ap.add_argument('-j', '--njobs', type=int, required=True,
                    help="Number of jobs/threads")
    args = ap.parse_args()
    n_estimators = args.ntrees
    n_jobs = args.njobs

    # Perform tests
    for dataset in datasets:
        out_folder = f'results_early_iris_{n_estimators}_trees/{dataset}'
        perform_curve_test_on_iris(
            dataset_name=dataset,
            partition=1,
            n_jobs=n_jobs,
            out_folder=out_folder,
            model_params={'n_estimators': n_estimators},
            verbose=1
        )


def main_feature_selection_bench():
    from benchmark_tests.load_benchmark_datasets import \
        load_partitions_cancer, load_dataset_h41, load_dataset_m41, \
        load_dataset_s51, load_partitions_higgs
    from model_tests import select_features_rf
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('n_jobs', type=int, help='Number of jobs/processes')
    args = ap.parse_args()
    n_jobs = args.n_jobs
    # Perform tests
    load_fns = [load_partitions_cancer, load_dataset_h41, load_dataset_m41,
                load_dataset_s51]
    names = ['cancer', 'h41', 'm41', 's51']
    params = [{'n_estimators': 50}, None, None, None]
    # load_fns = [load_partitions_cancer, load_dataset_h41, load_dataset_m41,
    #             load_dataset_s51, load_partitions_higgs]
    # names = ['cancer', 'h41', 'm41', 's51', 'higgs']
    # params = [{'n_estimators': 50}, None, None, None, {'n_estimators': 500}]
    root_folder = 'results_selection_rf/'
    for load_fn, name, param in zip(load_fns, names, params):
        select_features_rf(
            load_fn=load_fn,
            n_jobs=n_jobs,
            out_folder=root_folder + name,
            prepartitioned=False,
            model_params=param,
        )


def main_feature_selection_iris():
    from sklearn.model_selection import train_test_split
    from load_partitions import load_partitions
    from model_tests import select_features_rf
    from constants import datasets

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('n_jobs', type=int, help='Number of jobs/processes')
    args = ap.parse_args()
    n_jobs = args.n_jobs
    # Perform tests
    root_folder = 'results_selection_rf_iris/'
    for dataset in datasets:
        def load_iris_tvt():
            train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
                dataset, 1, 0, True
            )
            test_x, val_x, test_y, val_y = train_test_split(
                test_x, test_y, test_size=0.25, stratify=test_y)
            return train_x, train_y, val_x, val_y, test_x, test_y

        select_features_rf(
            load_fn=load_iris_tvt,
            n_jobs=n_jobs,
            out_folder=root_folder + dataset,
            prepartitioned=True,
            model_params={'n_estimators': 500},
            limit_features=200
        )


if __name__ == '__main__':
    main_feature_selection_iris()
