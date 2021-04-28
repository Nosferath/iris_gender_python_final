import numpy as np


def generate_subindexes(pairs):
    n_pairs = pairs.shape[0]
    assert n_pairs % 5 == 0, 'Number of pairs to generate subindexes must be' \
        ' divisible by 5.'
    np.random.seed(42)
    indexes = np.repeat([0, 1, 2, 3, 4], n_pairs / 5)
    np.random.shuffle(indexes)
    subindexes = np.zeros(n_pairs * 2, dtype=int)
    for i in range(n_pairs):
        cur_pair = pairs[i, :2].astype(int)
        subindexes[cur_pair] = indexes[i]
    return subindexes
