import pickle
from multiprocessing import Pool
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from constants import TEST_SIZE
from load_data import load_iris_dataset, load_dataset_both_eyes
from load_data_utils import partition_data, partition_both_eyes
from utils import Timer
from vgg import load_vgg_model_finetune, load_vgg_model_features, \
    prepare_data_for_vgg, labels_to_onehot, load_periocular_vgg, \
    load_periocular_pre_vgg, prepare_botheyes_for_vgg, \
    load_periocular_botheyes_vgg, labels_to_onehot_botheyes, \
    load_periocular_botheyes_pre_vgg


def vgg_feat_lsvm_parall(data_x, labels, partition: int):
    # Prepare data
    train_x, train_y, test_x, test_y = partition_data(
        data_x, labels, TEST_SIZE, partition
    )
    # Train model
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LinearSVC(max_iter=5000,
                            random_state=42))
    ])
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    results = classification_report(test_y, pred, output_dict=True)

    del train_x, train_y, test_x, test_y
    return results


def _perform_vgg_feat_lsvm_test(data, labels, dataset_name: str,
                                n_partitions: int, n_jobs: int, out_folder):
    """Performs VGG feat lsvm test on an already loaded dataset."""
    args = [(data, labels, i) for i in range(n_partitions)]
    with Pool(n_jobs) as p:
        print("VGG Features, LSVM Test")
        t = Timer(f"{dataset_name}, {n_partitions} partitions, {n_jobs} jobs")
        t.start()
        results = p.starmap(vgg_feat_lsvm_parall, args)
        t.stop()

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}.pickle', 'wb') as f:
        pickle.dump(results, f)


def perform_vgg_feat_lsvm_test(dataset_name: str, n_partitions: int,
                               n_jobs: int,
                               out_folder='vgg_feat_lsvm_results'):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    feat_model = load_vgg_model_features()
    data, labels = load_iris_dataset(dataset_name, None)
    data = prepare_data_for_vgg(data)
    data = feat_model.predict(data)
    t.stop()
    del feat_model
    _perform_vgg_feat_lsvm_test(data, labels, dataset_name, n_partitions,
                                n_jobs, out_folder)
    del data, labels


def perform_peri_vgg_feat_lsvm_test(eye: str, n_partitions: int, n_jobs: int,
                                    out_folder='vgg_feat_lsvm_peri_results'):
    t = Timer(f"Loading dataset VGG peri {eye}")
    t.start()
    data, labels, _ = load_periocular_vgg(eye)
    t.stop()
    _perform_vgg_feat_lsvm_test(data, labels, eye, n_partitions,
                                n_jobs, out_folder)
    del data, labels


def main_vgg_feat_lsvm_test():
    import argparse
    from constants import datasets

    ap = argparse.ArgumentParser()
    ap.add_argument('n_jobs', type=int,
                    help='Number of jobs')
    ap.add_argument('-p', '--n_parts', type=int, default=10,
                    help='Number of random partitions to test on')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    args = ap.parse_args()
    n_jobs = args.n_jobs
    n_parts = args.n_parts
    use_peri = args.use_peri

    if not use_peri:
        for d in datasets:
            perform_vgg_feat_lsvm_test(d, n_parts, n_jobs)
    else:
        for eye in ('left', 'right',):
            perform_peri_vgg_feat_lsvm_test(eye, n_parts, n_jobs)


def _perform_vgg_test(data, labels, dataset_name: str, partition: int,
                      out_folder):
    from tensorflow.keras.callbacks import TensorBoard
    results = []
    train_x, train_y, test_x, test_y = partition_data(
        data, labels, TEST_SIZE, partition
    )
    model = load_vgg_model_finetune()
    tb = TensorBoard(log_dir=f'vgg_logs/{dataset_name}/{partition}/',
                     write_graph=True, histogram_freq=0, write_images=True,
                     update_freq='batch')
    print("VGG Feats and Classifying Test")
    t = Timer(f"{dataset_name}, partition {partition}")
    t.start()
    model.fit(train_x, train_y, epochs=20, callbacks=[tb])
    preds = model.predict(test_x)
    preds = preds.argmax(axis=1)
    result = classification_report(test_y.argmax(axis=1), preds,
                                   output_dict=True)
    results.append(result)
    t.stop()

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}_{partition}.pickle', 'wb') as f:
        pickle.dump(results, f)

    del train_x, train_y, test_x, test_y


def perform_vgg_test(dataset_name: str, partition: int,
                     out_folder='vgg_full_results'):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    data, labels = load_iris_dataset(dataset_name, None)
    data = prepare_data_for_vgg(data)
    labels = labels_to_onehot(labels)
    t.stop()

    _perform_vgg_test(data, labels, dataset_name, partition, out_folder)
    del data, labels


def perform_peri_vgg_test(eye: str, partition: int,
                          out_folder='vgg_full_peri_results'):
    t = Timer(f"Loading dataset periocular pre-VGG {eye}")
    t.start()
    data, labels, _ = load_periocular_pre_vgg(eye)
    t.stop()

    _perform_vgg_test(data, labels, eye, partition, out_folder)
    del data, labels


def main_vgg_test():
    import argparse
    from constants import datasets

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--d_idx', type=int,
                    help='Dataset index')
    ap.add_argument('-p', '--n_part', type=int,
                    help='Number of partition to test on')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    args = ap.parse_args()
    d_idx = args.d_idx
    n_part = args.n_part
    use_peri = args.use_peri

    if not use_peri:
        d = datasets[d_idx]
        perform_vgg_test(d, n_part)
    else:
        eye = ('left', 'right')[d_idx]
        perform_peri_vgg_test(eye, n_part)


def vgg_feat_lsvm_parall_botheyes(all_data, males_set, females_set,
                                  partition: int, n_iters: int):
    # Prepare data
    train_x, train_y, test_x, test_y = partition_both_eyes(all_data, males_set,
                                                           females_set,
                                                           TEST_SIZE,
                                                           partition)
    # Train model
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LinearSVC(max_iter=n_iters,
                            random_state=42))
    ])
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    results = classification_report(test_y, pred, output_dict=True)

    del train_x, train_y, test_x, test_y
    return results


def _perform_vgg_feat_lsvm_test_botheyes(all_data, males_set, females_set,
                                         dataset_name: str, n_partitions: int,
                                         n_jobs, out_folder, n_iters):
    args = [(all_data, males_set, females_set, i, n_iters)
            for i in range(n_partitions)]
    with Pool(n_jobs) as p:
        print("VGG Features, LSVM Test , both eyes dataset")
        t = Timer(f"{dataset_name}, {n_partitions} partitions, {n_jobs} jobs")
        t.start()
        results = p.starmap(vgg_feat_lsvm_parall_botheyes, args)
        t.stop()

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}.pickle', 'wb') as f:
        pickle.dump(results, f)
    del all_data, males_set, females_set


def perform_vgg_feat_lsvm_test_botheyes(
        dataset_name: str, n_partitions: int, n_jobs: int,
        out_folder='vgg_feat_lsvm_botheyes_results', n_iters: int = 10000
):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    all_data, males_set, females_set = load_dataset_both_eyes(dataset_name)
    feat_model = load_vgg_model_features()
    all_data = prepare_botheyes_for_vgg(all_data)
    for eye in all_data:
        cur_x = all_data[eye][0]
        all_data[eye][0] = feat_model.predict(cur_x)
    t.stop()
    del feat_model
    _perform_vgg_feat_lsvm_test_botheyes(all_data, males_set, females_set,
                                         dataset_name, n_partitions, n_jobs,
                                         out_folder, n_iters)
    del all_data, males_set, females_set


def perform_vgg_feat_lsvm_test_botheyes_peri(
        n_partitions: int, n_jobs: int,
        out_folder='vgg_feat_lsvm_botheyes_peri_results',
        n_iters: int = 10000
):
    dataset_name = 'both_peri'
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    all_data, males_set, females_set = load_periocular_botheyes_vgg()
    t.stop()
    _perform_vgg_feat_lsvm_test_botheyes(all_data, males_set, females_set,
                                         dataset_name, n_partitions, n_jobs,
                                         out_folder, n_iters)
    del all_data, males_set, females_set


def main_vgg_feat_lsvm_test_botheyes():
    import argparse
    from constants import datasets_botheyes

    ap = argparse.ArgumentParser()
    ap.add_argument('n_jobs', type=int,
                    help='Number of jobs')
    ap.add_argument('-p', '--n_parts', type=int, default=10,
                    help='Number of random partitions to test on')
    ap.add_argument('-i', '--n_iters', type=int, default=10000,
                    help='Number of iterations for LSVM')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    args = ap.parse_args()
    n_jobs = args.n_jobs
    n_parts = args.n_parts
    n_iters = args.n_iters
    use_peri = args.use_peri

    if not use_peri:
        out_folder = f'vgg_feat_lsvm_botheyes_results_{n_iters/1000:0.0f}' \
                     f'k_iters'
        for d in datasets_botheyes:
            perform_vgg_feat_lsvm_test_botheyes(d, n_parts, n_jobs, out_folder,
                                                n_iters)
    else:
        out_folder = f'vgg_feat_lsvm_botheyes_peri_results_' \
                     f'{n_iters/1000:0.0f}k_iters'
        perform_vgg_feat_lsvm_test_botheyes_peri(n_parts, n_jobs, out_folder,
                                                 n_iters)


def _perform_vgg_test_botheyes(all_data, males_set, females_set,
                               dataset_name: str, partition: int,
                               out_folder, epochs=20):
    from tensorflow.keras.callbacks import TensorBoard
    results = []
    train_x, train_y, test_x, test_y = partition_both_eyes(all_data, males_set,
                                                           females_set,
                                                           TEST_SIZE,
                                                           partition)
    model = load_vgg_model_finetune()
    tb = TensorBoard(log_dir=f'vgg_logs_botheyes/{dataset_name}/{partition}/',
                     write_graph=True, histogram_freq=0, write_images=True,
                     update_freq='batch')
    print("VGG Feats and Classifying Test, Both Eyes")
    t = Timer(f"{dataset_name}, partition {partition}")
    t.start()
    model.fit(train_x, train_y, epochs=epochs, callbacks=[tb])
    preds = model.predict(test_x)
    preds = preds.argmax(axis=1)
    result = classification_report(test_y.argmax(axis=1), preds,
                                   output_dict=True)
    results.append(result)
    t.stop()

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}_{partition}.pickle', 'wb') as f:
        pickle.dump(results, f)

    del train_x, train_y, test_x, test_y


def perform_vgg_test_botheyes(dataset_name: str, partition: int,
                              out_folder='vgg_full_botheyes_results',
                              epochs=20):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    all_data, males_set, females_set = load_dataset_both_eyes(dataset_name)
    all_data = prepare_botheyes_for_vgg(all_data)
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, dataset_name,
                               partition, out_folder, epochs)
    del all_data, males_set, females_set


def perform_peri_vgg_test_botheyes(
        partition: int,
        out_folder='vgg_full_peri_botheyes_results',
        epochs=20
):
    eye = 'both_peri'
    t = Timer(f"Loading dataset periocular pre-VGG {eye}")
    t.start()
    all_data, males_set, females_set = load_periocular_botheyes_pre_vgg()
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, eye,
                               partition, out_folder, epochs)
    del all_data


def main_vgg_botheyes_test():
    import argparse
    from constants import datasets_botheyes

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--d_idx', type=int,
                    help='Dataset index')
    ap.add_argument('-p', '--n_part', type=int,
                    help='Number of partition to test on')
    ap.add_argument('-e', '--epochs', type=int, default=20,
                    help='Number of epochs to train for')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    args = ap.parse_args()
    d_idx = args.d_idx
    n_part = args.n_part
    use_peri = args.use_peri
    epochs = args.epochs

    if not use_peri:
        d = datasets_botheyes[d_idx]
        perform_vgg_test_botheyes(d, n_part, epochs=epochs)
    else:
        perform_peri_vgg_test_botheyes(n_part, epochs=epochs)


if __name__ == '__main__':
    # main_vgg_feat_lsvm_test()
    # main_vgg_test()
    # main_vgg_feat_lsvm_test_botheyes()
    main_vgg_botheyes_test()
