import numpy as np
from sklearn.ensemble import AdaBoostClassifier
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
                         pair_method: str):
    return find_best_params(train_x, train_y, dataset_name, partition,
                            folder_name, pair_method,
                            start_param_a=200, step_param_a=200,
                            end_param_a=1000, start_param_b=-5, step_param_b=1,
                            end_param_b=1,
                            param_a_islog2=False, param_b_islog2=True,
                            param_a_min1=True, param_b_min1=False,
                            param_grid_fn=get_ada_param_grid,
                            clasif_name='ADA', clasif_fn=AdaBoostClassifier)


def main(find_params=True):
    main_base(find_params=find_params,
              out_params_name='ada_params',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=False)


def main_std(find_params=True):
    main_base(find_params=find_params,
              out_params_name='ada_params_std',
              find_params_fn=find_best_ada_params,
              out_results_name='ada_results_std',
              clasif_fn=AdaBoostClassifier,
              use_std_masks=True)
