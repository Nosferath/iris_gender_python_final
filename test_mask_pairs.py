from pathlib import Path

import numpy as np

from constants import datasets_botheyes, FEMALES_LABEL, MALES_LABEL
from load_data_utils import partition_both_eyes, balance_partition
from load_data import load_dataset_both_eyes
from mask_pairs import generate_pairs, calculate_spp_matrix
from results_processing import analize_pairs, plot_pairs_histogram, \
    plot_pairs_analysis

BAD_SCORE = 0.1

partition = 1

out_folder = Path('experiments/mask_pairs/analysis')
out_folder.mkdir(exist_ok=True, parents=True)

for d in datasets_botheyes:
    all_data, males_set, females_set = load_dataset_both_eyes(d)
    train_data, _ = partition_both_eyes(
        all_data, males_set, females_set, 0.2, partition,
        apply_pairs=True, dataset_name=d, get_all_data_pairs=True
    )
    train_x, train_y, train_m, train_n = balance_partition(**train_data)
    spp_mat = calculate_spp_matrix(train_m[train_y == FEMALES_LABEL, :],
                                   train_m[train_y == MALES_LABEL, :])

    analysis = {}
    thresholds = list(np.arange(0.03, 0.151, 0.005))
    # thresholds = list(np.arange(0.05, 0.151, 0.005))
    for threshold in thresholds:
        train_pairs, pair_scores = generate_pairs(train_y, train_m,
                                                  threshold=threshold,
                                                  spp_mat=spp_mat)
        cur_analysis = analize_pairs(pair_scores, bad_score=BAD_SCORE)
        analysis[threshold] = cur_analysis

    avg_good_scores = [a['avg_good_score'] for a in analysis.values()]
    n_bad_pairs = [a['n_bad_pairs'] for a in analysis.values()]

    plot_pairs_analysis(thresholds, avg_good_scores,
                        n_bad_pairs, out_folder, d)

    # Generate histograms
    hist_folder = out_folder / 'histograms'
    hist_folder.mkdir(exist_ok=True, parents=True)
    for threshold, cur_analysis in analysis.items():
        hist = np.array(cur_analysis['histogram'])
        bins = cur_analysis['bins']
        plot_pairs_histogram(hist, bins, hist_folder, d, threshold, max_y=50)
