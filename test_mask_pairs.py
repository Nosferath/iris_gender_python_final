from constants import datasets_botheyes
from load_data_utils import partition_both_eyes
from load_data import load_dataset_both_eyes

for d in datasets_botheyes:
    all_data, males_set, females_set = load_dataset_both_eyes(d)
    _ = partition_both_eyes(all_data, males_set, females_set, 0.2, 1, True, d)
    