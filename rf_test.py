import numpy as np
from sklearn.ensemble import RandomForestClassifier

from constants import MODEL_PARAMS_FOLDER
from template_test import get_param_grid, find_best_params, main_base


def get_rf_param_grid(start_ntrees, step_ntrees, end_ntrees, start_maxfeats,
                      step_maxfeats, end_maxfeats):
    param_grid = get_param_grid(start_param_a=start_ntrees,
                                step_param_a=step_ntrees,
                                end_param_a=end_ntrees,
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


def find_best_rf_params(train_x: np.ndarray, train_y: np.ndarray,
                        dataset_name: str, partition: int, folder_name: str,
                        pair_method: str, n_jobs: int):
    sqrt_feats = np.round(np.sqrt(train_x.shape[1]))
    start_maxfeats = int(np.floor(sqrt_feats / 2.0))
    step_maxfeats = int(np.ceil(start_maxfeats / 2.0))
    end_maxfeats = start_maxfeats + 4 * step_maxfeats
    return find_best_params(train_x, train_y, dataset_name, partition,
                            folder_name, pair_method,
                            start_param_a=200, step_param_a=200,
                            end_param_a=2000, start_param_b=start_maxfeats,
                            step_param_b=step_maxfeats,
                            end_param_b=end_maxfeats,
                            param_a_islog2=False, param_b_islog2=False,
                            param_a_min1=True, param_b_min1=False,
                            param_grid_fn=get_rf_param_grid,
                            clasif_name='RF', clasif_fn=RandomForestClassifier,
                            n_jobs=n_jobs)


def main(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results',
              clasif_fn=RandomForestClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs)


def main_std(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params_std',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results_std',
              clasif_fn=RandomForestClassifier,
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
                              f'/rf_params_cmim_{n_cmim}',
              find_params_fn=find_best_rf_params,
              out_results_name=f'rf_results_cmim_{n_cmim}',
              clasif_fn=RandomForestClassifier,
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
                              f'/rf_params_std_cmim_{n_cmim}',
              find_params_fn=find_best_rf_params,
              out_results_name=f'rf_results_std_cmim_{n_cmim}',
              clasif_fn=RandomForestClassifier,
              use_std_masks=True,
              n_cmim=n_cmim,
              n_jobs=n_jobs)


def main_mod(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params_mod',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results_mod',
              clasif_fn=RandomForestClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod)


def main_std_mod(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params_std_mod',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results_std_mod',
              clasif_fn=RandomForestClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod)


def main_mod_v2(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod_v2
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params_mod_v2',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results_mod_v2',
              clasif_fn=RandomForestClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod_v2)


def main_std_mod_v2(find_params=True, n_jobs=-1):
    from load_partitions import load_partitions_cmim_mod_v2
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/rf_params_std_mod_v2',
              find_params_fn=find_best_rf_params,
              out_results_name='rf_results_std_mod_v2',
              clasif_fn=RandomForestClassifier,
              use_std_masks=True,
              n_cmim=0,
              n_jobs=n_jobs,
              dataset_loading_fn=load_partitions_cmim_mod_v2)


if __name__ == '__main__':
    main()
