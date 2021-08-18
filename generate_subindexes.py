import numpy as np
from sklearn.model_selection import KFold


def generate_subindexes(pairs):
    """Generate partition subindexes that are based on pairs, so that
    elements of the same pair are always in the same partition.
    (For Step 9)

    THIS FUNCTION ASSUMES PAIRS HAVE BEEN REINDEXED

    The returned array contains the fold index for each element.
    """
    n_pairs = pairs.shape[0]
    kfold = KFold(n_splits=5, shuffle=False, random_state=42)
    subindexes = np.full(2 * n_pairs, -1, dtype=int)
    for fold_idx, fold in enumerate(kfold.split(np.arange(n_pairs))):
        _, cur_fold = fold
        for pair_idx in cur_fold:
            cur_pair = pairs[pair_idx, :]
            subindexes[cur_pair] = fold_idx
    assert -1 not in subindexes, "Not all elements were set to a fold"
    return subindexes
