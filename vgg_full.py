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
    from tensorflow import convert_to_tensor
    results = []
    train_x, train_y, test_x, test_y = partition_data(
        data, labels, TEST_SIZE, partition
    )
    train_x_t = convert_to_tensor(train_x)
    train_y_t = convert_to_tensor(train_y)
    test_x_t = convert_to_tensor(test_x)
    model = load_vgg_model_finetune()
    tb = TensorBoard(log_dir=f'vgg_logs/{dataset_name}/{partition}/',
                     write_graph=True, histogram_freq=0, write_images=True,
                     update_freq='batch')
    print("VGG Feats and Classifying Test")
    t = Timer(f"{dataset_name}, partition {partition}")
    t.start()
    model.fit(train_x_t, train_y_t, epochs=20, callbacks=[tb])
    preds = model.predict(test_x_t)
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
                               dataset_name: str, partition: int, params: dict,
                               out_folder, step_by_step=False):
    from tensorflow.keras import backend as K
    from tensorflow import convert_to_tensor
    epochs = params['epochs']
    use_val = params['use_val']
    lr = params['learning_rate']
    batch_size = params['batch_size']
    architecture = params['architecture']

    def eval_model(mdl, _train, _test, _val):
        _results = {}
        for _name, _data in (('train', _train),
                             ('test', _test),
                             ('val', _val)):
            if _data is None:
                continue
            _x, _y = _data
            _y = _y.argmax(axis=1)
            _preds = mdl.predict(convert_to_tensor(_x),
                                 batch_size=batch_size).argmax(axis=1)
            print(f'{_name} results: \n', classification_report(_y, _preds))
            _results[_name] = classification_report(_y, _preds,
                                                    output_dict=True)
        return _results

    out_folder = Path(out_folder)
    results = []
    train_x, train_y, test_x, test_y = partition_both_eyes(all_data, males_set,
                                                           females_set,
                                                           TEST_SIZE,
                                                           partition)
    train_x_t = convert_to_tensor(train_x)
    train_y_t = convert_to_tensor(train_y)
    test_x_t = convert_to_tensor(test_x)
    test_y_t = convert_to_tensor(test_y)
    if dataset_name.startswith('240') or dataset_name.startswith('480'):
        use_peri = False
    else:
        use_peri = True
    if not use_peri:
        # Change input shape to closest possible if dataset is normalized
        w = int(dataset_name[:3])
        h = int(dataset_name[4:6])
        input_shape = (max(h, 32), w, 3)
        model = load_vgg_model_finetune(lr=lr, input_shape=input_shape,
                                        architecture=architecture)
    else:
        model = load_vgg_model_finetune(lr=lr, architecture=architecture)
    print("VGG Feats and Classifying Test, Both Eyes")
    if not use_val:
        val_data = (test_x_t, test_y_t)
    else:
        from sklearn.model_selection import train_test_split
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y,
                                                        test_size=0.5,
                                                        stratify=test_y)
        test_x_t = convert_to_tensor(test_x)
        test_y_t = convert_to_tensor(test_y)
        val_x_t = convert_to_tensor(val_x)
        val_y_t = convert_to_tensor(val_y)
        val_data = (val_x_t, val_y_t)
    if not step_by_step and not use_peri:
        from tensorflow.keras.callbacks import TensorBoard
        tb = TensorBoard(
            log_dir=f'{out_folder}/logs/{dataset_name}/{partition}/',
            write_graph=True, histogram_freq=0, write_images=True,
            update_freq='batch'
        )
        t = Timer(f"{dataset_name}, partition {partition}")
        t.start()
        model.fit(train_x_t, train_y_t, epochs=epochs, callbacks=[tb],
                  validation_data=val_data, batch_size=batch_size)
        preds = model.predict(test_x_t, batch_size=batch_size)
        preds = preds.argmax(axis=1)
        result = classification_report(test_y.argmax(axis=1), preds,
                                       output_dict=True)
        results.append(result)
        t.stop()
    elif step_by_step and not use_peri:
        train = (train_x, train_y)
        test = (test_x, test_y)
        val = val_data if use_val else None
        result = eval_model(model, train, test, val)
        results.append(result)
        for _ in range(epochs):
            model.fit(train_x_t, train_y_t, epochs=1,
                      validation_data=val_data, batch_size=batch_size)
            result = eval_model(model, train, test, val)
            results.append(result)
            # prompt = input('Press <ENTER> to continue, or type stop to stop')
            # if prompt.lower() == 'stop':
            #     break
    else:
        from vgg_callback import EvaluateCallback
        if use_val:
            val_data = ((train_x, train_y), (test_x, test_y), val_data)
        else:
            val_data = ((train_x, train_y), (test_x, test_y))
        cb = EvaluateCallback(val_data,
                              out_folder / f'{dataset_name}_{partition}',
                              batch_size)
        t = Timer(f"{dataset_name}, partition {partition}")
        t.start()
        model.fit(train_x_t, train_y_t, epochs=epochs, batch_size=batch_size,
                  callbacks=[cb])
        preds = model.predict(test_x_t, batch_size=batch_size)
        preds = preds.argmax(axis=1)
        result = classification_report(test_y.argmax(axis=1), preds,
                                       output_dict=True)
        results.append(result)
        t.stop()

    out_folder.mkdir(exist_ok=True, parents=True)
    with open(out_folder / f'{dataset_name}_{partition}.pickle', 'wb') as f:
        pickle.dump(results, f)

    K.clear_session()
    del train_x, train_y, test_x, test_y, model


def perform_vgg_test_botheyes(
        dataset_name: str,
        partition: int,
        params: dict,
        out_folder='vgg_full_botheyes_results',
        step_by_step=False
):
    t = Timer(f"Loading dataset {dataset_name}")
    t.start()
    all_data, males_set, females_set = load_dataset_both_eyes(dataset_name)
    all_data = prepare_botheyes_for_vgg(all_data, preserve_shape=True)
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, dataset_name,
                               partition, params=params, out_folder=out_folder,
                               step_by_step=step_by_step)
    del all_data, males_set, females_set


def perform_peri_vgg_test_botheyes(
        partition: int,
        params: dict,
        out_folder='vgg_full_peri_botheyes_results',
        step_by_step=False
):
    eye = 'both_peri'
    t = Timer(f"Loading dataset periocular pre-VGG {eye}")
    t.start()
    all_data, males_set, females_set = load_periocular_botheyes_pre_vgg()
    all_data = labels_to_onehot_botheyes(all_data)
    t.stop()

    _perform_vgg_test_botheyes(all_data, males_set, females_set, eye,
                               partition, params=params, out_folder=out_folder,
                               step_by_step=step_by_step)
    del all_data


def main_vgg_botheyes_test():
    import argparse
    import json
    from constants import datasets_botheyes

    ap = argparse.ArgumentParser()
    ap.add_argument('-pf', '--params_file', type=str,
                    help='.json file with parameters')
    ap.add_argument('-d', '--d_idx', type=int,
                    help='Dataset index')
    ap.add_argument('-p', '--n_part', type=int,
                    help='Number of partition to test on')
    ap.add_argument('--use_peri', action='store_true',
                    help='Perform periocular test')
    ap.add_argument('-o', '--out_folder', type=str, default=None,
                    help='Folder where results and logs will be output')
    ap.add_argument('-sbs', '--step_by_step', action='store_true',
                    help='Supervise training step by step')
    args = ap.parse_args()
    params_file = Path(args.params_file)
    with open(params_file, 'r') as f:
        params = json.load(f)
    d_idx = args.d_idx
    n_part = args.n_part
    use_peri = args.use_peri
    out_folder = args.out_folder
    step_by_step = args.step_by_step

    if out_folder is None:
        out_folder = str(params_file.parent)

    if not use_peri:
        d = datasets_botheyes[d_idx]
        perform_vgg_test_botheyes(d, n_part, params=params,
                                  out_folder=out_folder,
                                  step_by_step=step_by_step)
    else:
        perform_peri_vgg_test_botheyes(n_part, params=params,
                                       out_folder=out_folder,
                                       step_by_step=step_by_step)


if __name__ == "__main__":
    main_vgg_botheyes_test()
