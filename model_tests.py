import pickle
from pathlib import PurePath, Path
from typing import Callable, Union

import numpy as np


def partition_dataset(load_fn: Callable):
    from sklearn.model_selection import train_test_split
    data_x, data_y = load_fn()
    data_y[data_y == -1] = 0
    test_prop = 0.3  # Proporion of data_x to be used as test+val
    val_prop = 1 / 3  # Proportion of test to be used as val
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=test_prop, stratify=data_y
    )
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=val_prop, stratify=test_y
    )
    return train_x, train_y, val_x, val_y, test_x, test_y


def training_curve_test(load_fn: Callable, n_jobs: int,
                        out_folder: Union[str, PurePath],
                        prepartitioned: bool,
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


def select_features_xgboost(load_fn: Callable, n_jobs: int,
                            out_folder: Union[str, PurePath],
                            prepartitioned: bool,
                            model_params: Union[None, dict],
                            skip_last: bool = False,
                            limit_features: int = 0):
    """Trains and evaluates XGBoost as a feature selector and as a
    classifier. Does not use validation, so val and test are merged.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import classification_report
    import xgboost as xgb
    # Load data
    if prepartitioned:
        train_x, train_y, val_x, val_y, test_x, test_y = load_fn()
    else:
        # Load and split data
        train_x, train_y, val_x, val_y, test_x, test_y = partition_dataset(
            load_fn
        )
    # Generate selector model
    if model_params is None:
        model_params = {}
    sel_model = xgb.XGBClassifier(
        **model_params,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=n_jobs
    )
    sel_model.fit(
        train_x, train_y,
    )
    pred = sel_model.predict(test_x)
    report = classification_report(test_y, pred, output_dict=True)
    report['n_feats'] = train_x.shape[1]
    results = {}
    results_early = {}
    # Iterate over the model thresholds
    thresholds = np.sort(sel_model.feature_importances_)
    if limit_features:
        thresholds = thresholds[:min(len(thresholds), limit_features)]
    for thresh in thresholds:
        # Select features
        selector = SelectFromModel(sel_model, threshold=thresh, prefit=True)
        sel_train_x = selector.transform(train_x)
        sel_val_x = selector.transform(val_x)
        sel_test_x = selector.transform(test_x)
        # Train model
        model = xgb.XGBClassifier(
            **model_params,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=n_jobs,
        )
        model.fit(
            sel_train_x, train_y,
            eval_set=[(sel_train_x, train_y), (sel_val_x, val_y)],
            eval_metric=['error', 'logloss'], verbose=0
        )
        # Evaluate model
        pred = model.predict(sel_test_x)
        report = classification_report(test_y, pred, output_dict=True)
        report['n_feats'] = sel_train_x.shape[1]
        results[thresh] = report

        model.fit(
            sel_train_x, train_y,
            eval_set=[(sel_train_x, train_y), (sel_val_x, val_y)],
            eval_metric=['error', 'logloss'], early_stopping_rounds=10,
            verbose=0
        )
        pred = model.predict(sel_test_x)
        report = classification_report(test_y, pred, output_dict=True)
        report['n_feats'] = sel_train_x.shape[1]
        results_early[thresh] = report
    # Save results
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    results_name = out_folder.name + '_results.pickle'
    with open(out_folder / results_name, 'wb') as f:
        pickle.dump(results, f)
    results_name = out_folder.name + '_results_early.pickle'
    with open(out_folder / results_name, 'wb') as f:
        pickle.dump(results_early, f)
    # return results, results_early
    # Generate plots
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # Generate DataFrames and remove unnecesary data
    if skip_last:
        del results[0], results_early[0]
    df = pd.DataFrame(results)
    df = df.drop(['0', '1', 'macro avg', 'weighted avg'], axis=0)
    df = df.transpose()
    df_early = pd.DataFrame(results_early)
    df_early = df_early.drop(['0', '1', 'macro avg', 'weighted avg'], axis=0)
    df_early = df_early.transpose()
    # Rename columns and merge
    df.columns = ['normal', 'n_feats']
    df.index.name = 'fscore threshold'
    df_early.columns = ['early', 'n_feats']
    df_early.index.name = 'fscore threshold'
    df = pd.merge(df, df_early, on=['fscore threshold', 'n_feats'])
    df = df.reset_index()
    # Melt df for plotting
    df = df.melt(
        id_vars=['n_feats', 'fscore threshold'],
        value_vars=['normal', 'early'],
        var_name='test type', value_name='accuracy'
    )
    sns.set_style('whitegrid')
    sns.set_context('talk')
    ax = sns.lineplot(data=df, x='n_feats', y='accuracy', hue='test type')
    plt.legend()
    ax.set_xlabel('N. Features')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(f'Feature selection, XGBoost, dataset '
                 f'{out_folder.name.upper()}')
    plot_name = f'{out_folder.name}_plot.png'
    plt.tight_layout()
    plt.savefig(out_folder / plot_name)
    plt.clf()
    return results, results_early
