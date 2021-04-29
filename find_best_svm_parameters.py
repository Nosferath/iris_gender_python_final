from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC

from generate_subindexes import generate_subindexes
from pairs import load_pairs_array
from utils import Timer


def get_svm_param_grid(start_log2c, step_log2c, end_log2c, start_log2g,
                       step_log2g, end_log2g):
    nsteps_log2c = np.floor((end_log2c - start_log2c + 1) / step_log2c)
    nsteps_log2g = np.floor((end_log2g - start_log2g + 1) / step_log2g)
    c = np.logspace(start_log2c, end_log2c, nsteps_log2c, base=2)
    g = np.logspace(start_log2g, end_log2g, nsteps_log2g, base=2)
    param_grid = {'C': c, 'gamma': g, 'kernel': ['rbf']}
    return param_grid


def find_best_svm_params(train_x, train_y, dataset_name, out_folder_name):
    # Create out folder
    out_folder = Path.cwd() / out_folder_name
    out_folder.mkdir(exist_ok=True)
    # Initialize param. grid
    start_log2c = -2
    end_log2c = 10
    step_log2c = 1
    start_log2g = 4
    step_log2g = 0.5
    end_log2g = 12
    param_grid = get_svm_param_grid(start_log2c, step_log2c, end_log2c,
                                    start_log2g, step_log2g, end_log2g)
    # Get subindexes
    pairs = load_pairs_array(dataset_name)
    subindexes = generate_subindexes(pairs)  # TODO include pairsParts?
    # First CV
    t = Timer("SVM CV1 execution time:")
    svm = GridSearchCV(SVC(), param_grid, n_jobs=-1,
                       cv=PredefinedSplit(subindexes), verbose=1)
    t.start()
    svm.fit(train_x, train_y)
    t.stop()
    # TODO implement second CV
