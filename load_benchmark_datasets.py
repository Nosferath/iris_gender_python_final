import numpy as np
import pandas as pd


def load_partitions_cancer(dataset_name: str, partition: int,
                           mask_value: float, scale_dataset: bool,
                           pair_method: str, n_cmim: int):
    """Wrapper function (implementing the same interface as
    load_partitions_cmim) for loading the breast_cancer dataset.
    Used for testing that XGBoost is working as intended, through the
    same pipeline used by Iris dataset.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=partition)
    return x_train, y_train, None, None, x_test, y_test, None, None


def load_partitions_higgs(dataset_name: str = None, partition: int = None,
                          mask_value: float = None, scale_dataset: bool = None,
                          pair_method: str = None, n_cmim: int = None):
    """Wrapper function (implementing the same interface as
    load_partitions_cmim) for loading the breast_cancer dataset.
    Used for testing that XGBoost is working as intended, through the
    same pipeline used by Iris dataset.
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("_additional_xgb_tests/HIGGS.csv", header=None)
    data_arr = df.to_numpy()
    data_x = data_arr[:, 1:]
    data_y = data_arr[:, 0]
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=1000000, random_state=42)
    return train_x, train_y, None, None, test_x, test_y, None, None


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


def load_dataset_s51():
    from Bio import SeqIO

    with open(r"_additional_xgb_tests\M6AMRFS-master\Dataset-S51.fasta") as f:
        sequences = list(SeqIO.parse(f, 'fasta'))
    n_seq = len(sequences)
    seqs: list = [None] * n_seq
    labels: list = [None] * n_seq
    for i in range(n_seq):
        cur_seq = sequences[i]
        seqs[i] = np.hstack([
            encode_dinucleotids(str(cur_seq.seq)),
            lpsdf(str(cur_seq.seq))
        ])
        labels[i] = int(cur_seq.description.split('|')[-1])
    data_x = np.array(seqs)
    data_y = np.array(labels)
    return data_x, data_y
