from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from results_processing import generate_training_curves
from utils import generate_dmatrix


DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}
SEED = 42


def phase_1(data: dict, lr_list, out_folder, njobs: int, data_name: str):
    """Adjust learning rate and number of trees."""
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # Determine initial value for number of trees
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=1 / 3, stratify=test_y, random_state=SEED
    )
    results = {}
    # Get base number of trees from xgb native CV
    for lr in lr_list:
        cur_results = {}
        model = XGBClassifier(
            **DEFAULT_PARAMS,
            learning_rate=lr,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            nthread=njobs,
            scale_pos_weight=1,
            seed=SEED
        )
        dmatrix = generate_dmatrix(train_x, train_y)
        params = model.get_params()
        nbr = params['n_estimators']
        cv_results = xgb.cv(
            params,
            dmatrix,
            num_boost_round=nbr,
            metrics='logloss',
            early_stopping_rounds=50,
            nfold=5,
            verbose_eval=True
        )
        cur_results['cv_results'] = cv_results
        n_est = cv_results.shape[0]
        # Evaluate using these parameters
        model.set_params(
            n_estimators=n_est,
        )
        model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (val_x, val_y)],
            eval_metric=['error', 'logloss']
        )
        pred = model.predict(test_x)
        report = classification_report(test_y, pred, output_dict=True)
        cur_results['report'] = report
        results[lr] = cur_results
        generate_training_curves(
            model, out_folder / f'phase1_{lr:.02f}/curves',
            f'XGBoost ({data_name})')

    with open(out_folder / 'phase1_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    # Explore other values for n_estimators
    param_grid = []
    for lr in lr_list:
        base_n_est = results[lr]['cv_results'].shape[0]
        base_n_est -= base_n_est % 10  # Round down
        n_est_list = np.arange(base_n_est - 20, base_n_est + 21, 10)
        param_grid.append({'learning_rate': [lr],
                           'n_estimators': n_est_list})
    model = XGBClassifier(
        **DEFAULT_PARAMS,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        seed=SEED
    )
    model = GridSearchCV(model, param_grid, n_jobs=njobs, cv=5,
                         return_train_score=True)
    full_x = np.vstack([train_x, test_x])
    full_y = np.hstack([train_y, test_y])
    model.fit(full_x, full_y)

    with open(out_folder / 'phase1_cvmodel.pickle', 'wb') as f:
        pickle.dump(model, f)

    return results, model


def generate_phase_1_gridplot(cv_results, out_file):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(cv_results)
    df = df[['param_n_estimators', 'param_learning_rate', 'mean_test_score']]
    df.param_learning_rate = df.param_learning_rate.astype(float).round(2)
    df.param_n_estimators = df.param_n_estimators.astype(int)
    df.mean_test_score = df.mean_test_score.astype(float)*100
    df = df.pivot(index='param_learning_rate', columns='param_n_estimators',
                  values='mean_test_score')
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(df, cmap='jet', ax=ax, annot=True, fmt='.1f')
        ax.set_title(f'Phase 1 parameter grid, {out_file.parent.name}')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.clf()


def phase_2(cv_results):
    idx = np.where(cv_results['rank_test_score'] == 1)[0][0]
    params = cv_results['params'][idx]

