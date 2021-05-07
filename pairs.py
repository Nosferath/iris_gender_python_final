from scipy.io import loadmat


def load_pairs_array(dataset_name: str):
    """Loads the pairs array from the .mat file. The array is fixed
    so it is zero-indexed for use with Python and numpy."""
    pairs = loadmat('pairs/' + dataset_name + '.mat')
    pairs = pairs['pairs']
    pairs[:, [0, 1]] = pairs[:, [0, 1]] - 1
    return pairs
