import pickle
from pathlib import Path, PurePath
from typing import Union, Callable

import numpy as np


def get_nucleotids_results(model):
    rank = model.cv_results_['rank_test_accuracy']
    best_idx = np.where(rank == 1)[0][0]
    best_acc = model.cv_results_['mean_test_accuracy'][best_idx]
    best_sn = model.cv_results_['mean_test_sensitivity'][best_idx]
    best_sp = model.cv_results_['mean_test_specificity'][best_idx]
    best_mcc = model.cv_results_['mean_test_mcc'][best_idx]
    results = {
        'accuracy': best_acc,
        'sensitivity': best_sn,
        'specificity': best_sp,
        'mcc': best_mcc
    }
    return results


def ready_nucleotids_model(n_jobs: int, verbose: int = 1):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, matthews_corrcoef
    import xgboost as xgb
    params = {
        'n_estimators': np.linspace(1, 10, 10, dtype=int),
        'eta': np.linspace(0.1, 0.8, 8),
        'max_depth': np.linspace(2, 10, 9, dtype=int),
        'objective': ['binary:logistic'],
        'use_label_encoder': [False],
        'eval_metric': ['logloss'],
        'tree_method': ['gpu_hist']
    }
    scoring = {
        'accuracy': 'accuracy',
        'sensitivity': 'recall',
        'specificity': 'precision',
        'mcc': make_scorer(matthews_corrcoef)
    }
    model = GridSearchCV(xgb.XGBClassifier(),
                         param_grid=params,
                         scoring=scoring,
                         cv=10,
                         refit=False,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         error_score='raise')

    return model


def evaluate_nucleotids_model(data_x: np.ndarray, data_y: np.ndarray,
                              n_jobs: int, out_file: Union[str, Path] = None,
                              verbose: int = 2):
    """Performs the same evaluations done by the nucleotids paper."""
    data_y[data_y == -1] = 0
    model = ready_nucleotids_model(n_jobs=n_jobs, verbose=verbose)

    model.fit(data_x, data_y)
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, 'wb') as f:
            pickle.dump(model, f)
    results = get_nucleotids_results(model)
    print(results)
    return results


def evaluate_nucleotids_model_fs(data_x: np.ndarray, data_y: np.ndarray,
                                 n_jobs: int,
                                 out_file: Union[str, Path] = None,
                                 verbose: int = 2):
    """Performs the same evaluations done by the nucleotids paper."""
    from benchmark_tests.feature_selection import fscore
    data_y[data_y == -1] = 0
    rank = fscore(data_x, data_y, exclude_nan=True)
    model = ready_nucleotids_model(n_jobs=n_jobs, verbose=verbose)
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
    all_results = {}
    for i in range(len(rank)):
        cur_idx = rank[:i + 1]
        cur_data_x = data_x[:, cur_idx]
        model.fit(cur_data_x, data_y)
        results = get_nucleotids_results(model)
        all_results[i] = results
        if out_file is not None:
            name = out_file.stem + f'_{i}' + out_file.suffix
            cur_out = out_file.parent / name
            with open(cur_out, 'wb') as f:
                pickle.dump(model, f)
            results_name = out_file.stem + '_results' + out_file.suffix
            cur_out = out_file.parent / results_name
            with open(cur_out, 'wb') as f:
                pickle.dump(all_results, f)

    return all_results


def main_nucleotids(load_fn: Callable, out_file: Union[str, Path], n_jobs: int,
                    verbose: int = 2):
    data_x, data_y = load_fn()
    evaluate_nucleotids_model(data_x=data_x, data_y=data_y, n_jobs=n_jobs,
                              out_file=out_file, verbose=verbose)


def main_nucleotids_fs(load_fn: Callable, out_file: Union[str, Path],
                       n_jobs: int, verbose: int = 2):
    """Performs nucleotids test using feature selection."""
    data_x, data_y = load_fn()
    return evaluate_nucleotids_model_fs(data_x=data_x, data_y=data_y,
                                        n_jobs=n_jobs, out_file=out_file,
                                        verbose=verbose)


def training_curve_test(load_fn: Callable, n_jobs: int,
                        out_folder: Union[str, PurePath], verbose: int = 1):
    """Performs a 'training curve test' in which a validation partition
    is used to monitor training. Uses the XGBoost model.

    Partitions are separated into train:val:test with a 70:10:20 ratio.
    """
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from results_processing import generate_training_curves
    # Create out folder
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # Load and split data
    seed = np.random.MT19937().random_raw()
    data_x, data_y = load_fn()
    data_y[data_y == -1] = 0
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.3, stratify=data_y
    )
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=1 / 3, stratify=test_y
    )
    # Prepare and train model
    for_values = (
        ('curves', 'full_training', {}),
        ('curves_early', 'early_stop', {'early_stopping_rounds': 10})
    )
    for curves_name, results_name, early_arg in for_values[0:1]:
        # First loop trains fully and generates full curves
        # Second loop does early stopping
        model = XGBClassifier(
            n_estimators=1000,  # TODO borrar o poner opcional
            # eta=0.5,
            # max_depth=4,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=n_jobs,
            random_state=seed
        )
        model.fit(train_x, train_y, **early_arg,
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  eval_metric=['error', 'logloss'],
                  verbose=verbose)
        # Evaluate model and generate plots
        curves_file = out_folder / curves_name
        generate_training_curves(model, curves_file, 'XGBoost')
        results_file = out_folder / f'results_{results_name}.pickle'
        preds = model.predict(test_x)
        results = classification_report(test_y, preds, output_dict=True)
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
