from scipy.io import loadmat


def load_pairs_array(dataset_name: str):
    pairs = loadmat('pairs/' + dataset_name + '.mat')
    pairs = pairs['pairs']
    pairs[:, [0, 1]] = pairs[:, [0, 1]] - 1
    return pairs
