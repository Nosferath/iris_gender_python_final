import pickle
from pathlib import Path

from sklearn.metrics import classification_report

from constants import TEST_SIZE
from load_data import load_iris_dataset, load_dataset_both_eyes
from load_data_utils import partition_data, partition_both_eyes
from utils import Timer
from vgg_utils import load_vgg_model_finetune, prepare_data_for_vgg, \
    labels_to_onehot, load_periocular_pre_vgg, prepare_botheyes_for_vgg, \
    labels_to_onehot_botheyes, load_periocular_botheyes_pre_vgg


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


def _perform_vgg_test_botheyes(all_data, males_set, females_set,
                               dataset_name: str, partition: int,
                               out_folder, epochs=20, use_val=False, lr=0.001,
                               batch_size=32):
    from tensorflow.keras.callbacks import TensorBoard
    results = []
    train_x, train_y, test_x, test_y = partition_both_eyes(all_data, males_set,
                                                           females_set,
                                                           TEST_SIZE,
                                                           partition)
    if dataset_name.startswith('240') or dataset_name.startswith('480'):
        # Change input shape to closest possible if dataset is normalized
        w = int(dataset_name[:3])
        h = int(dataset_name[4:6])
        input_shape = (max(h, 32), w, 3)
        model = load_vgg_model_finetune(lr=lr, input_shape=input_shape)
    else:
        model = load_vgg_model_finetune(lr=lr, use_newfc2=False)
    tb = TensorBoard(log_dir=f'{out_folder}/logs/{dataset_name}/{partition}/',
                     write_graph=True, histogram_freq=0, write_images=True,
                     update_freq='batch')
    print("VGG Feats and Classifying Test, Both Eyes")
    t = Timer(f"{dataset_name}, partition {partition}")
    t.start()
    if not use_val:
        model.fit(train_x, train_y, epochs=epochs, callbacks=[tb],
                  validation_data=(test_x, test_y), batch_size=batch_size)
    else:
        from sklearn.model_selection import train_test_split
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y,
                                                        test_size=0.5,
                                                        stratify=test_y)
        model.fit(train_x, train_y, epochs=epochs, callbacks=[tb],
                  validation_data=(val_x, val_y), batch_size=batch_size)

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
                              epochs=21, use_val=False, lr=0.001,
                              batch_size=32):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    all_data, males_set, females_set = load_dataset_both_eyes(dataset_name)
    all_data = prepare_botheyes_for_vgg(all_data, preserve_shape=True)
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, dataset_name,
                               partition, out_folder, epochs, use_val, lr=lr,
                               batch_size=batch_size)
    del all_data, males_set, females_set


def perform_peri_vgg_test_botheyes(
        partition: int,
        out_folder='vgg_full_peri_botheyes_results',
        epochs=20,
        use_val=False,
        lr=0.001,
        batch_size=32
):
    eye = 'both_peri'
    t = Timer(f"Loading dataset periocular pre-VGG {eye}")
    t.start()
    all_data, males_set, females_set = load_periocular_botheyes_pre_vgg()
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, eye,
                               partition, out_folder, epochs, use_val, lr=lr,
                               batch_size=batch_size)
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
    ap.add_argument('--use_val', action='store_true',
                    help='Use a separate validation set')
    ap.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Learning rate for training')
    ap.add_argument('-bs', '--batch_size', type=int, default=32,
                    help='Batch size for training')
    ap.add_argument('-o', '--out_folder', type=str, default=None,
                    help='Folder where results and logs will be output')
    args = ap.parse_args()
    d_idx = args.d_idx
    n_part = args.n_part
    use_peri = args.use_peri
    epochs = args.epochs
    use_val = args.use_val
    lr = args.learning_rate
    batch_size = args.batch_size
    out_folder = args.out_folder

    if not use_peri:
        if out_folder is None:
            out_folder = 'vgg_full_botheyes_results'
        d = datasets_botheyes[d_idx]
        perform_vgg_test_botheyes(d, n_part, out_folder=out_folder,
                                  epochs=epochs, use_val=use_val,
                                  lr=lr, batch_size=batch_size)
    else:
        if out_folder is None:
            out_folder = 'vgg_full_peri_botheyes_results'
        perform_peri_vgg_test_botheyes(n_part, out_folder=out_folder,
                                       epochs=epochs, use_val=use_val,
                                       lr=lr, batch_size=batch_size)


if __name__ == "__main__":
    main_vgg_botheyes_test()
