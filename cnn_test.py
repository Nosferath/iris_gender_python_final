import argparse
from itertools import product
from pathlib import Path

# from tensorflow.keras.backend import set_session, clear_session, get_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf

from load_partitions import load_partitions
from pairs import load_pairs_array
from standard_masks import generate_standard_masks, apply_std_mask


def reset_keras(model):
    # sess = get_session()
    # clear_session()
    # sess.close()
    # sess = get_session()
    #
    # try:
    #     del model  # this is from global space - change this as you need
    # except:
    #     pass
    #
    # # use the same config as you used to create the session
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    # config.gpu_options.visible_device_list = "0"
    # set_session(tf.Session(config=config))
    pass


def prepare_x_array(x_array, orig_shape):
    return x_array.reshape((-1, orig_shape[0], orig_shape[1], 1))


def make_cnn_model(orig_shape: tuple, dropout: bool):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=4, activation='relu',
                     input_shape=(orig_shape[0], orig_shape[1], 1)))
    if dropout:
        model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=4, activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=4, activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(1536, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train_cnn(train_x, train_y, test_x, test_y, orig_shape: tuple,
              dropout: bool):
    # Prepare data
    train_x = prepare_x_array(train_x, orig_shape)
    test_x = prepare_x_array(test_x, orig_shape)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    # Create model and add layers
    # model = Sequential()
    # model.add(Conv2D(16, kernel_size=4, activation='relu',
    #                  input_shape=(orig_shape[0], orig_shape[1], 1)))
    # if dropout:
    #     model.add(Dropout(0.3))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Conv2D(32, kernel_size=4, activation='relu'))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, kernel_size=4, activation='relu'))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # if dropout:
    #     model.add(Dropout(0.3))
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # if dropout:
    #     model.add(Dropout(0.5))
    # model.add(Dense(1536, activation='relu'))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # print(model.summary())
    # Train model
    model = make_cnn_model(orig_shape, dropout)
    model.fit(train_x, train_y, batch_size=32, epochs=100,
              validation_data=(test_x, test_y))
    # Evaluate model
    scores = model.evaluate(test_x, test_y)
    return model, scores


def make_simple_cnn_model(orig_shape: tuple, dropout: bool):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=4, activation='relu',
                     input_shape=(orig_shape[0], orig_shape[1], 1)))
    if dropout:
        model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=4, activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=4, activation='relu'))
    if dropout:
        model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train_cnn_simpler(train_x, train_y, test_x, test_y, orig_shape: tuple,
                      dropout: bool):
    # Prepare data
    train_x = prepare_x_array(train_x, orig_shape)
    test_x = prepare_x_array(test_x, orig_shape)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    # Create model and add layers
    # model = Sequential()
    # model.add(Conv2D(8, kernel_size=4, activation='relu',
    #                  input_shape=(orig_shape[0], orig_shape[1], 1)))
    # if dropout:
    #     model.add(Dropout(0.3))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Conv2D(16, kernel_size=4, activation='relu'))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Conv2D(32, kernel_size=4, activation='relu'))
    # if dropout:
    #     model.add(Dropout(0.3))
    # model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # if dropout:
    #     model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # print(model.summary())
    # Train model
    model = make_simple_cnn_model(orig_shape, dropout)
    model.fit(train_x, train_y, batch_size=32, epochs=100,
              validation_data=(test_x, test_y))
    # Evaluate model
    scores = model.evaluate(test_x, test_y)
    return model, scores


def full_test(use_std_masks: bool = False):
    # datasets = ('left_240x20', 'right_240x20', 'left_240x40', 'right_240x40',
    #             'left_480x80', 'right_480x80')
    datasets = ('left_240x40', 'right_240x40', 'left_480x80', 'right_480x80')
    if use_std_masks:
        results_folder = Path.cwd() / 'cnn_results_std'
    else:
        results_folder = Path.cwd() / 'cnn_results'
    results_folder.mkdir(exist_ok=True)
    results_file = results_folder / 'full_results.csv'
    if results_file.exists():
        results = pd.read_csv(results_file, index_col=0)
    else:
        results = pd.DataFrame(columns=['database', 'dataset', 'partition',
                                        'method', 'accuracy'])
    for dataset, part in product(datasets, range(1, 11)):
        train_x, train_y, train_m, _, test_x, test_y, _, _ = load_partitions(
            dataset, part, mask_value=0, scale_dataset=True)
        if use_std_masks:
            pairs = load_pairs_array(dataset)
            train_m = generate_standard_masks(train_m, pairs)
            train_x = apply_std_mask(train_x, train_m, mask_value=0)
        orig_shape = (dataset.split('_')[-1]).split('x')
        orig_shape = (int(orig_shape[1]), int(orig_shape[0]))
        for simpler, dropout in product((True, False), (True, False)):
            method = "cnn"
            out_folder = "cnn_models"
            if dropout:
                method += "_dropout"
                out_folder += "_dropout"
            if simpler:
                method += "_simpler"
                out_folder += "_simpler"
            if use_std_masks:
                out_folder += "_std"
            print("Current dataset: {}, partition {}. Method: {}".format(
                dataset, part, method))
            condition = (results.dataset == dataset) & \
                        (results.partition == part) & \
                        (results.method == method)
            if len(results[condition]) > 0:
                print("Test already performed. Skipping.")
                continue
            out_folder = Path(out_folder)
            out_folder.mkdir(exist_ok=True)
            if simpler:
                model, scores = train_cnn_simpler(
                    train_x, train_y, test_x, test_y, orig_shape, dropout)
            else:
                model, scores = train_cnn(
                    train_x, train_y, test_x, test_y, orig_shape, dropout)
            print("Accuracy: %.2f%%" % (scores[1]*100))
            # Save model
            model.save(out_folder / (dataset + '_' + str(part) + '.h5'))
            cur_results = {
                'database': 'UND_GFI',
                'dataset': dataset,
                'partition': part,
                'method': method,
                'accuracy': scores[1] * 100
            }
            results = results.append(cur_results, ignore_index=True)
            results.to_csv(results_folder / 'full_results.csv')
            reset_keras(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset-name', type=str,
                    help='Name of the dataset to test on')
    ap.add_argument('out-folder', type=str,
                    help='Name of the folder to save the model')
    ap.add_argument('-p', '--partition', default=1, type=int,
                    choices=range(1, 11),
                    help='Partition number, from 1 to 10')
    ap.add_argument('--dropout', action='store_true',
                    help='Include dropout in the model')
    ap.add_argument('--simpler', action='store_true',
                    help='Use a simpler model')
    # Parse arguments
    args = vars(ap.parse_args())
    dataset_name = args['dataset-name']
    orig_shape = (dataset_name.split('_')[-1]).split('x')
    orig_shape = (int(orig_shape[1]), int(orig_shape[0]))
    partition = args['partition']
    out_folder = Path(args['out-folder'])
    out_folder.mkdir(exist_ok=True)
    dropout = args['dropout']
    simpler = args['simpler']
    # Load data
    train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
        dataset_name, partition, mask_value=0, scale_dataset=True)
    # Train and test model
    if simpler:
        model, scores = train_cnn_simpler(
            train_x, train_y, test_x, test_y, orig_shape, dropout)
    else:
        model, scores = train_cnn(train_x, train_y, test_x, test_y, orig_shape,
                                  dropout)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    # Save model
    model.save(out_folder / (dataset_name + '_' + str(partition) + '.h5'))


if __name__ == '__main__':
    # main()
    full_test(use_std_masks=True)
