import numpy as np


def fscore(data_x: np.ndarray, data_y: np.ndarray, exclude_nan: bool):
    """Returns a vector with the ranks of features based on F-score."""
    from sklearn.feature_selection import SelectKBest, f_classif
    fs = SelectKBest(f_classif, k='all')
    fs.fit(data_x, data_y)
    scores = fs.scores_
    rank = np.argsort(scores)[::-1]
    # Put nan at the end
    scores = scores[rank]
    nan_idx = np.where(np.isnan(scores))[0].max()
    if exclude_nan:
        return rank[nan_idx + 1:]
    rank = np.hstack([rank[nan_idx + 1:], rank[:nan_idx + 1]])
    return rank