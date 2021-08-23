from sklearn.ensemble import BaggingClassifier

from template_test import get_param_grid


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
