import numpy as np
import xgboost as xgb

from constants import MODEL_PARAMS_FOLDER
from template_test import get_param_grid, find_best_params, main_base, \
    find_best_params_xgb, main_base_xgb


XGB_INIT_PARAMS = {'objective': 'binary:logistic',
                   'use_label_encoder': False,
                   'eval_metric': 'logloss',
                   'tree_method': 'gpu_hist'}


def xgb_demo():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                              use_label_encoder=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    return model


def get_xgb_param_grid(start_nestimators, step_nestimators, end_nestimators,
                       start_maxdepth, step_maxdepth, end_maxdepth):
    param_grid = get_param_grid(start_param_a=start_nestimators,
                                step_param_a=step_nestimators,
                                end_param_a=end_nestimators,
                                start_param_b=start_maxdepth,
                                step_param_b=step_maxdepth,
                                end_param_b=end_maxdepth,
                                param_a_name='n_estimators',
                                param_b_name='max_depth',
                                param_a_func='linspace',
                                param_b_func='linspace',
                                type_a=int,
                                type_b=int)
    return param_grid


def find_best_xgb_params(train_x: np.ndarray, train_y: np.ndarray,
                         dataset_name: str, partition: int, folder_name: str,
                         pair_method: str, n_jobs: int):
    params = find_best_params(train_x, train_y, dataset_name, partition,
                              folder_name, pair_method,
                              start_param_a=40, step_param_a=20,
                              end_param_a=100, start_param_b=2,
                              step_param_b=2,
                              end_param_b=6,
                              param_a_islog2=False, param_b_islog2=False,
                              param_a_min1=True, param_b_min1=False,
                              param_grid_fn=get_xgb_param_grid,
                              clasif_name='XGB', clasif_fn=xgb.XGBClassifier,
                              n_jobs=n_jobs, init_params=XGB_INIT_PARAMS)
    return params


def find_best_xgb_params_early(train_x: np.ndarray, train_y: np.ndarray,
                               val_x: np.ndarray, val_y: np.ndarray,
                               dataset_name: str, partition: int,
                               folder_name: str,
                               pair_method: str, n_jobs: int):
    params = find_best_params_xgb(
        train_x, train_y, val_x, val_y, dataset_name, partition,
        folder_name, pair_method,
        start_param_a=40, step_param_a=20,
        end_param_a=100, start_param_b=2,
        step_param_b=2,
        end_param_b=6,
        param_a_islog2=False, param_b_islog2=False,
        param_a_min1=True, param_b_min1=False,
        param_grid_fn=get_xgb_param_grid,
        clasif_name='XGB', clasif_fn=xgb.XGBClassifier,
        n_jobs=n_jobs, init_params=XGB_INIT_PARAMS)
    return params


def find_best_xgb_params_cancer(train_x: np.ndarray, train_y: np.ndarray,
                                dataset_name: str, partition: int,
                                folder_name: str, pair_method: str,
                                n_jobs: int):
    params = find_best_params(train_x, train_y, dataset_name, partition,
                              folder_name, pair_method,
                              start_param_a=40, step_param_a=20,
                              end_param_a=100, start_param_b=2,
                              step_param_b=2,
                              end_param_b=6,
                              param_a_islog2=False, param_b_islog2=False,
                              param_a_min1=True, param_b_min1=True,
                              param_grid_fn=get_xgb_param_grid,
                              clasif_name='XGB', clasif_fn=xgb.XGBClassifier,
                              n_jobs=n_jobs, init_params=XGB_INIT_PARAMS,
                              no_pairs=True)
    return params


def main(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/xgb_params',
              find_params_fn=find_best_xgb_params,
              out_results_name='xgb_results',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              init_params=XGB_INIT_PARAMS)


def main_cmim_early(n_cmim: int, find_params=True, n_jobs=-1):
    main_base_xgb(find_params=find_params,
                  out_params_name=MODEL_PARAMS_FOLDER +
                                  f'/xgb_params_cmim_{n_cmim}_early',
                  find_params_fn=find_best_xgb_params_early,
                  out_results_name=f'xgb_results_cmim_{n_cmim}_early',
                  clasif_fn=xgb.XGBClassifier,
                  use_std_masks=False,
                  n_cmim=n_cmim,
                  n_jobs=n_jobs,
                  init_params=XGB_INIT_PARAMS)


def main_cancer(find_params=True, n_jobs=-1):
    """This version uses the cancer dataset for testing performance."""
    from load_benchmark_datasets import load_partitions_cancer
    main_base(find_params=find_params,
              out_params_name='cancer_params/xgb_params',
              find_params_fn=find_best_xgb_params_cancer,
              out_results_name='cancer_results/xgb_results',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cancer,
              init_params=XGB_INIT_PARAMS)


def main_higgs(find_params=True, n_jobs=-1):
    """This version uses the Higgs dataset for testing performance."""
    from load_benchmark_datasets import load_partitions_higgs
    main_base(find_params=find_params,
              out_params_name='noparams',
              find_params_fn=lambda *x, **y: {},
              out_results_name='_additional_xgb_tests/xgb_higgs_results',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_higgs,
              init_params={**XGB_INIT_PARAMS, 'n_estimators': 500})


def main_lbps(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_lbps
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/xgb_lbps_r3_noparams',
              find_params_fn=lambda *x, **y: {},  # find_best_xgb_params,
              out_results_name='xgb_lbps_r3_results_noparams',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_lbps,
              init_params=XGB_INIT_PARAMS)


def main_hlbps(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_hlbps
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/xgb_hlbps_r1_noparams_mask',
              find_params_fn=lambda *x, **y: {},  # find_best_xgb_params,
              out_results_name='xgb_hlbps_r1_results_noparams_mask',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_hlbps,
              init_params=XGB_INIT_PARAMS)
