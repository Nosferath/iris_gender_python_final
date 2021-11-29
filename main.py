import argparse
from pathlib import PurePath
from typing import Union

from benchmark_tests.model_tests import training_curve_test
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


def main_iris(n_estimators: int, n_jobs: int):
    from constants import datasets
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--ntrees', type=int, required=True,
                    help="Number of estimators")
    ap.add_argument('-j', '--njobs', type=int, required=True,
                    help="Number of jobs/threads")
    args = ap.parse_args()
    ntrees = args.ntrees
    njobs = args.njobs
    main_iris(n_estimators=ntrees, n_jobs=njobs)


if __name__ == '__main__':
    main()
