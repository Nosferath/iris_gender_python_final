from pathlib import Path, PurePath
from typing import Union

import numpy as np


def load_partitions_cancer():
    """Wrapper function for loading the breast_cancer dataset. """
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    data_x = cancer.data
    data_y = cancer.target
    return data_x, data_y


def load_partitions_higgs():
    """Wrapper function (implementing the same interface as
    load_partitions_cmim) for loading the breast_cancer dataset.
    Used for testing that XGBoost is working as intended, through the
    same pipeline used by Iris dataset.
    """
    import pandas as pd
    # from sklearn.model_selection import train_test_split
    df = pd.read_csv(Path.cwd() / "_additional_xgb_tests/HIGGS.csv",
                     header=None)
    data_arr = df.to_numpy()
    data_x = data_arr[:, 1:]
    data_y = data_arr[:, 0]
    # train_x, test_x, train_y, test_y = train_test_split(
    #     data_x, data_y, test_size=1000000, random_state=42)
    # return train_x, train_y, test_x, test_y
    return data_x, data_y


def encode_dinucleotids(in_seq: str):
    """Binary encoding of overlapping dinucleotids"""
    from itertools import product
    if 'U' in in_seq:
        bases = 'AUCG'
    else:
        bases = 'ATCG'
    codes = {
        ba+bb: [int(j) for j in f'{value:04b}']
        for value, (ba, bb) in enumerate(product(bases, bases))
    }
    len_seq = len(in_seq)
    encoded = []
    for j in range(len_seq - 1):
        dinucleotid = in_seq[j:j+2]
        encoded.extend(codes[dinucleotid])

    return np.array(encoded)


def lpsdf(in_seq: str):
    """Local Position-Specific Dinucleotide Frequency"""
    from itertools import product
    if 'U' in in_seq:
        bases = 'AUCG'
    else:
        bases = 'ATCG'
    counter = {
        ba+bb: 0 for value, (ba, bb) in enumerate(product(bases, bases))
    }
    len_seq = len(in_seq)
    feat_vector = []
    for j in range(len_seq - 1):
        dinucleotid = in_seq[j:j+2]
        counter[dinucleotid] += 1
        freq = counter[dinucleotid] / (j + 2)
        feat_vector.append(freq)

    return np.array(feat_vector)


def load_dataset_nucleotids(filename: Union[str, PurePath], has_labels: bool):
    from Bio import SeqIO
    with open(Path(filename)) as f:
        sequences = list(SeqIO.parse(f, 'fasta'))
    n_seq = len(sequences)
    seqs = []
    labels = []
    for i in range(n_seq):
        cur_seq = sequences[i]
        # seqs[i] = np.hstack([
        #     encode_dinucleotids(str(cur_seq.seq)),
        #     lpsdf(str(cur_seq.seq))
        # ])
        try:
            seqs.append(encode_dinucleotids(str(cur_seq.seq)))
        except KeyError:
            continue
        if has_labels:
            labels.append(int(cur_seq.description.split('|')[-1]))
        else:
            labels.append(-1 if i >= n_seq / 2 else 1)

    data_x = np.array(seqs)
    data_y = np.array(labels)
    return data_x, data_y


def load_dataset_s51():
    filename = "_additional_xgb_tests/M6AMRFS-master/Dataset-S51.fasta"
    filename = Path.cwd() / filename
    return load_dataset_nucleotids(filename, has_labels=True)


def load_dataset_h41():
    filename = "_additional_xgb_tests/M6AMRFS-master/Dataset-H41.fasta"
    filename = Path.cwd() / filename
    return load_dataset_nucleotids(filename, has_labels=False)


def load_dataset_m41():
    filename = "_additional_xgb_tests/M6AMRFS-master/Dataset-M41.fasta"
    filename = Path.cwd() / filename
    return load_dataset_nucleotids(filename, has_labels=False)


def load_dataset_a101():
    filename = "_additional_xgb_tests/M6AMRFS-master/Dataset-A101.fasta"
    filename = Path.cwd() / filename
    data_x, data_y = load_dataset_nucleotids(filename, has_labels=False)
    unbalance = np.sum(data_y)
    if unbalance > 0:
        data_x = data_x[unbalance:, :]
        data_y = data_y[unbalance:]
    elif unbalance < 0:
        data_x = data_x[:unbalance, :]
        data_y = data_y[:unbalance]
    assert np.sum(data_y) == 0, 'Unbalance not solved'
    return data_x, data_y


def generate_random_dataset(n_examples: int, n_features: int):
    data_x = np.random.random((n_examples, n_features))
    data_y = np.zeros(n_examples, dtype=int)
    ones = np.random.choice(n_examples, n_examples//2, replace=False)
    data_y[ones] = 1
    return data_x, data_y
