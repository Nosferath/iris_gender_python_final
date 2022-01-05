import pickle
from multiprocessing import Pool
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from constants import TEST_SIZE
from load_data import load_iris_dataset
from load_data_utils import partition_data
from utils import Timer
from vgg import load_vgg_model_finetune, load_vgg_model_features, \
    prepare_data_for_vgg, labels_to_onehot, load_periocular_vgg


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
    _perform_vgg_feat_lsvm_test(data, labels, dataset_name, n_partitions,
                                n_jobs, out_folder)
    del feat_model, data, labels


def perform_peri_vgg_feat_lsvm_test(eye: str, n_partitions: int, n_jobs: int,
                                    out_folder='vgg_feat_lsvm_peri_results'):
    t = Timer(f"Loading dataset VGG peri {eye}")
    t.start()
    data, labels = load_periocular_vgg(eye)
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


def perform_vgg_test(dataset_name: str, n_partitions: int,
                     n_jobs: int):
    from tensorflow.keras.callbacks import TensorBoard
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    data, labels = load_iris_dataset(dataset_name, None)
    data = prepare_data_for_vgg(data)
    labels = labels_to_onehot(labels)
    t.stop()

    results = []
    for part in range(n_partitions):
        train_x, train_y, test_x, test_y = partition_data(
            data, labels, 0.3, part
        )
        model = load_vgg_model_finetune()
        tb = TensorBoard(log_dir='vgg_logs/', write_graph=True, )
        print("VGG Feats and Classifying Test")
        t = Timer(f"{dataset_name}, {n_partitions} partitions, {n_jobs} jobs")
        t.start()
        model.fit(train_x, train_y, epochs=5)
        preds = model.predict(test_x)
        preds = preds.argmax(axis=1)
        result = classification_report(test_y.argmax(axis=1), preds,
                                       output_dict=True)
        results.append(result)
        t.stop()

    out_folder = Path('vgg_feat_lsvm_results_prelim')
    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}.pickle', 'wb') as f:
        pickle.dump(results, f)

    del data, labels


def main_vgg_test():
    pass


if __name__ == '__main__':
    main_vgg_feat_lsvm_test()
