from scipy.io import loadmat


def load_cmim_array(dataset_name: str):
    cmim_array = loadmat('cmimArrays/' + dataset_name + '.mat')
    cmim_array = cmim_array['cmimArray']
    return cmim_array
