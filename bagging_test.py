import numpy as np
from sklearn.ensemble import BaggingClassifier

from template_test import get_param_grid, find_best_params, main_base


def get_bag_param_grid(start_nestimators, step_nestimators, end_nestimators,
                       start_maxfeats, step_maxfeats, end_maxfeats):
    param_grid = get_param_grid(start_param_a=start_nestimators,
                                step_param_a=step_nestimators,
                                end_param_a=end_nestimators,
                                start_param_b=start_maxfeats,
                                step_param_b=step_maxfeats,
                                end_param_b=end_maxfeats,
                                param_a_name='n_estimators',
                                param_b_name='max_features',
                                param_a_func='linspace',
                                param_b_func='linspace',
                                type_a=int,
                                type_b=int)
    return param_grid


def find_best_bag_params(train_x: np.ndarray, train_y: np.ndarray,
                         dataset_name: str, partition: int, folder_name: str,
                         pair_method: str, n_jobs: int):
    start_maxfeats = int(train_x.shape[1] / 2)
    step_maxfeats = int((train_x.shape[1] - start_maxfeats)/5)
    start_maxfeats = start_maxfeats - step_maxfeats
    end_maxfeats = train_x.shape[1] - step_maxfeats
    return find_best_params(train_x, train_y, dataset_name, partition,
                            folder_name, pair_method,
                            start_param_a=100, step_param_a=50,
                            end_param_a=500, start_param_b=start_maxfeats,
                            step_param_b=step_maxfeats,
                            end_param_b=end_maxfeats,
                            param_a_islog2=False, param_b_islog2=False,
                            param_a_min1=True, param_b_min1=True,
                            param_grid_fn=get_bag_param_grid,
                            clasif_name='BAG', clasif_fn=BaggingClassifier,
                            n_jobs=n_jobs)


def main(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name='bag_params',
              find_params_fn=find_best_bag_params,
              out_results_name='bag_results',
              clasif_fn=BaggingClassifier,
              use_std_masks=False,
              n_jobs=n_jobs)


def main_std(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name='bag_params_std',
              find_params_fn=find_best_bag_params,
              out_results_name='bag_results_std',
              clasif_fn=BaggingClassifier,
              use_std_masks=True,
              n_jobs=n_jobs)
