import numpy as np


def get_param_grid(start_param_a, step_param_a, end_param_a,
                   start_param_b, step_param_b, end_param_b,
                   param_a_name, param_b_name,
                   param_a_func, param_b_func,
                   type_a, type_b):
    """Base function for the get_x_param_grid functions."""
    nsteps_a = int(np.floor((end_param_a - start_param_a + step_param_a)
                            / step_param_a))
    nsteps_b = int(np.floor((end_param_b - start_param_b + step_param_b)
                            / step_param_b))

    def get_param(start_param, end_param, nsteps, type_param, param_func):
        if param_func == 'linspace':
            param = np.linspace(start_param, end_param, nsteps,
                                dtype=type_param)
        elif param_func == 'logspace2':
            param = np.logspace(start_param, end_param, nsteps, base=2,
                                dtype=type_a)
        else:
            raise NotImplemented('This function has not been implemented for'
                                 'params.')
        return param

    param_a = get_param(start_param_a, end_param_a, nsteps_a, type_a,
                        param_a_func)
    param_b = get_param(start_param_b, end_param_b, nsteps_b, type_b,
                        param_b_func)
    param_grid = {param_a_name: param_a,
                  param_b_name: param_b}
    return param_grid
