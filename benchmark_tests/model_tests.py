import pickle
from pathlib import PurePath, Path
from typing import Callable, Union

import numpy as np


def partition_dataset(load_fn: Callable):
    from sklearn.model_selection import train_test_split
    data_x, data_y = load_fn()
    data_y[data_y == -1] = 0
    test_prop = 0.3  # Proporion of data_x to be used as test+val
    val_prop = 1/3  # Proportion of test to be used as val
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=test_prop, stratify=data_y
    )
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=val_prop, stratify=test_y
    )
    return train_x, train_y, val_x, val_y, test_x, test_y


def training_curve_test(load_fn: Callable, n_jobs: int,
                        out_folder: Union[str, PurePath],
                        prepartitioned: bool = False,
                        model_params: Union[None, dict] = None,
                        verbose: int = 1):
    """Performs a 'training curve test' in which a validation partition
    is used to monitor training. Uses the XGBoost model.

    Partitions are separated into train:val:test with a 70:10:20 ratio.
    """
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier
    from results_processing import generate_training_curves
    # Create out folder
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # Load data
    if prepartitioned:
        train_x, train_y, val_x, val_y, test_x, test_y = load_fn()
    else:
        # Load and split data
        train_x, train_y, val_x, val_y, test_x, test_y = partition_dataset(
            load_fn
        )
    # Prepare and train model
    seed = np.random.MT19937().random_raw()
    if model_params is None:
        model_params = {}
    for_values = (
        ('curves', 'full_training', {}),
        ('curves_early', 'early_stop', {'early_stopping_rounds': 10})
    )
    for curves_name, results_name, early_arg in for_values[0:1]:
        # First loop trains fully and generates full curves
        # Second loop does early stopping
        model = XGBClassifier(
            **model_params,
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
