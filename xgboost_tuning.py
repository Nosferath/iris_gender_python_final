from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from results_processing import generate_training_curves
from utils import generate_dmatrix, Timer


DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}
SEED = 42


def phase_1(data: dict, lr_list, out_folder, n_jobs: int, data_name: str):
    """Adjust learning rate and number of trees."""
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # Determine initial value for number of trees
    # train_x = data['train_x']
    # train_y = data['train_y']
    # test_x = data['test_x']
    # test_y = data['test_y']
    # test_x, val_x, test_y, val_y = train_test_split(
    #     test_x, test_y, test_size=1 / 3, stratify=test_y, random_state=SEED
    # )
    data_x = data['data_x']
    data_y = data['data_y']
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.3, stratify=data_y, random_state=SEED
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
            nthread=n_jobs,
            scale_pos_weight=1,
            seed=SEED
        )
        dmatrix = generate_dmatrix(train_x, train_y)
        params = model.get_params()
        nbr = params['n_estimators']
        timer = Timer(f'{data_name} initial results')
        timer.start()
        cv_results = xgb.cv(
            params,
            dmatrix,
            num_boost_round=nbr,
            metrics='logloss',
            early_stopping_rounds=50,
            nfold=5,
            verbose_eval=False
        )
        timer.stop()
        cur_results['initial_results'] = cv_results
        n_est = cv_results.shape[0]
        # Evaluate using these parameters
        model.set_params(
            n_estimators=n_est,
        )
        timer = Timer(f'{data_name} initial report')
        timer.start()
        model.fit(
            train_x, train_y,
            # eval_set=[(train_x, train_y), (val_x, val_y)],
            # eval_metric=['error', 'logloss']
        )
        pred = model.predict(test_x)
        timer.stop()
        report = classification_report(test_y, pred, output_dict=True)
        cur_results['initial_report'] = report
        results[lr] = cur_results
        # generate_training_curves(
        #     model, out_folder / f'phase1_{lr:.03f}/curves',
        #     f'XGBoost ({data_name})')

    with open(out_folder / 'phase1_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    # Explore other values for n_estimators
    param_grid = []
    for lr in lr_list:
        base_n_est = results[lr]['initial_results'].shape[0]
        base_n_est -= base_n_est % 10  # Round down
        base_n_est = max(base_n_est, 10)
        low = base_n_est - 20
        top = base_n_est + 21
        n_vals = (top - low - 1) / 10 + 1
        while n_vals < 11:
            low -= 10
            top += 20
            n_vals = (top - low - 1) / 10 + 1
        if low <= 0:
            top -= low + 10
            low = 10
        n_est_list = np.arange(low, top, 10)
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
    model = GridSearchCV(model, param_grid, n_jobs=n_jobs, cv=5,
                         return_train_score=True)
    # full_x = np.vstack([train_x, test_x])
    # full_y = np.hstack([train_y, test_y])
    # model.fit(full_x, full_y)
    timer = Timer(f'{data_name} CV phase 1')
    timer.start()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    timer.stop()
    results['cv_results'] = model.cv_results_
    results['cv_test'] = classification_report(test_y, pred, return_dict=True)

    with open(out_folder / 'phase1_cvmodel.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open(out_folder / 'phase1_results.pickle', 'wb') as f:
        pickle.dump(results, f)

    return results, model


def generate_phase_1_gridplot(cv_results, out_file, param_a='learning_rate',
                              param_b='n_estimators', param_a_round=True,
                              param_b_round=False, phase_n=1,
                              figsize=(16, 8)):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(cv_results)
    param_a = f'param_{param_a}'
    param_b = f'param_{param_b}'
    title = f'Phase {phase_n} parameter grid'
    df = df[[param_a, param_b, 'mean_test_score']]
    if param_a_round:
        df.loc[:, param_a] = df.loc[:, param_a].astype(float).round(3)
    else:
        df.loc[:, param_a] = df.loc[:, param_a].astype(int)
    if param_b_round:
        df.loc[:, param_b] = df.loc[:, param_b].astype(float).round(3)
    else:
        df.loc[:, param_b] = df.loc[:, param_b].astype(int)
    df.mean_test_score = df.mean_test_score.astype(float)*100
    df = df.drop_duplicates(subset=[param_a, param_b])
    df = df.pivot(index=param_a, columns=param_b,
                  values='mean_test_score')
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df, cmap='jet', ax=ax, annot=True, fmt='.1f')
        ax.set_title(f'{title}, {out_file.parent.name}')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.clf()


def phase_2(data, cv_results, out_folder, n_jobs: int, data_name: str):
    """Adjust max_depth and min_child_weight"""
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    idx = np.where(cv_results['rank_test_score'] == 1)[0][0]
    params = cv_results['params'][idx]

    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']

    model = XGBClassifier(
        **DEFAULT_PARAMS,
        **params,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        seed=SEED
    )
    param_grid = {
        'max_depth': range(3, 10, 1),
        'min_child_weight': list(np.arange(0, 1, 0.2)).append(
            list(range(1, 6, 1)))
    }

    model = GridSearchCV(model, param_grid, n_jobs=n_jobs, cv=5,
                         return_train_score=True)
    timer = Timer(f'{data_name} CV2')
    timer.start()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    timer.stop()
    results = {'cv_results': model.cv_results_,
               'cv_test': classification_report(test_y, pred)}

    with open(out_folder / 'phase2_results.pickle', 'wb') as f:
        pickle.dump(results, f)

    return results, model


## THESE ARE FOR CURRENT USE AND SHOULD BE DELETED SOON
# from constants import datasets
# from xgboost_tuning import generate_phase_1_gridplot
#
# root_path = 'results_xgb_params/'
# import pickle
#
# for d in ('M41', 'H41'):
#     with open(f'{root_path}{d}/phase2_results.pickle', 'rb') as f:
#         model = pickle.load(f)
#     out_file = f'{root_path}{d}/phase2_params.png'
#     generate_phase_1_gridplot(model['cv_results'], out_file,
#                               param_a='max_depth', param_b='min_child_weight',
#                               param_a_round=False, param_b_round=False,
#                               figsize=(8, 8))
#

# from constants import datasets
# from xgboost_tuning import generate_phase_1_gridplot
#
# root_path = 'results_xgb_params/'
# import pickle
#
# for d in ('M41',):
#     with open(f'{root_path}{d}/phase1_cvmodel.pickle', 'rb') as f:
#         model = pickle.load(f)
#     out_file = f'{root_path}{d}/phase1_params.png'
#     generate_phase_1_gridplot(model.cv_results_, out_file,
#                               param_a='n_estimators', param_b='learning_rate',
#                               param_a_round=False, param_b_round=True,
#                               figsize=(16, 14))
