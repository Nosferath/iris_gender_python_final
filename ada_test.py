import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from constants import MODEL_PARAMS_FOLDER
from template_test import get_param_grid, find_best_params, main_base


def get_ada_param_grid(start_nestimators, step_nestimators, end_nestimators,
                       startlog2_lr, steplog2_lr, endlog2_lr):
    param_grid = get_param_grid(start_param_a=start_nestimators,
                                step_param_a=step_nestimators,
                                end_param_a=end_nestimators,
                                start_param_b=startlog2_lr,
                                step_param_b=steplog2_lr,
                                end_param_b=endlog2_lr,
                                param_a_name='n_estimators',
                                param_b_name='learning_rate',
                                param_a_func='linspace',
                                param_b_func='logspace2',
                                type_a=int,
                                type_b='float64')
    return param_grid


def find_best_ada_params(train_x: np.ndarray, train_y: np.ndarray,
                         dataset_name: str, partition: int, folder_name: str,
                         pair_method: str, n_jobs: int):
    return find_best_params(train_x, train_y, dataset_name, partition,
                            folder_name, pair_method,
                            start_param_a=20, step_param_a=20,
                            end_param_a=100, start_param_b=-4, step_param_b=1,
                            end_param_b=1,
                            param_a_islog2=False, param_b_islog2=True,
                            param_a_min1=True, param_b_min1=False,
                            param_grid_fn=get_ada_param_grid,
                            clasif_name='ADA', clasif_fn=AdaBoostClassifier,
                            n_jobs=n_jobs)


def main(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs)


def main_std(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_std',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_std',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs)


def main_cmim(n_cmim: int, find_params=True, n_jobs=-1):
    """About n_cmim: When using CMIM, one must set the number of fea-
    tures to use. Currently, I am separating the features into 8 groups,
    and testing with the first 2 or 4 groups, as these groups have less
    masked features. That 2 or 4 would be the n_cmim parameter.
    """
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER +
                              f'/ada_params_cmim_{n_cmim}',
              find_params_fn=find_best_ada_params,
              out_results_name=f'ada_results_cmim_{n_cmim}',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False,
              n_cmim=n_cmim,
              n_jobs=n_jobs)


def main_std_cmim(n_cmim: int, find_params=True, n_jobs=-1):
    """About n_cmim: When using CMIM, one must set the number of fea-
    tures to use. Currently, I am separating the features into 8 groups,
    and testing with the first 2 or 4 groups, as these groups have less
    masked features. That 2 or 4 would be the n_cmim parameter.
    """
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER +
                              f'/ada_params_std_cmim_{n_cmim}',
              find_params_fn=find_best_ada_params,
              out_results_name=f'ada_results_std_cmim_{n_cmim}',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True,
              n_cmim=n_cmim,
              n_jobs=n_jobs)


def main_mod(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_mod',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_mod',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod)


def main_std_mod(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_std_mod',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_std_mod',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod)


def main_mod_v2(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod_v2
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_mod_v2',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_mod_v2',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod_v2)


def main_std_mod_v2(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod_v2
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_std_mod_v2',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_std_mod_v2',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod_v2)


def check_feature_importance(n_jobs):
    main_base(find_params=False,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_importance_p',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              check_feat_rank=True,
              do_double_feat_sort=True,
              permut_import=True)


def check_feature_importance_std(n_jobs):
    main_base(find_params=False,
              out_params_name=MODEL_PARAMS_FOLDER + '/ada_params_std',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_importance_std_p',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs,
              check_feat_rank=True,
              do_double_feat_sort=True,
              permut_import=True)
