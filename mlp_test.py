from itertools import product
from pathlib import Path

from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# from keras.utils import to_categorical
import numpy as np
import pandas as pd

from cnn_test import reset_keras
from load_partitions import load_partitions


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.05)


def train_mlp(train_x, train_y, test_x, test_y, model_idx: int):
    model_params = ((20, ), (40, ), (300, 40), (300, 80), (600, 80), (300, 80, 20))
    # Prepare data
    n_features = train_x.shape[1]
    # train_x = train_x.reshape((-1, n_features, 1))
    # test_x = test_x.reshape((-1, n_features, 1))
    # Create model and add layers
    model = Sequential()
    model.add(Dense(int(n_features * 1.5) if n_features < 5000 else 5000,
                    activation='tanh', input_dim=n_features))
    model.add(Dense(model_params[model_idx][0], activation='tanh'))
    if model_idx >= 2:
        model.add(Dense(model_params[model_idx][1], activation='tanh'))
    if model_idx >= 5:
        model.add(Dense(model_params[model_idx][2], activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=SGD(lr=0.0001), loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    # Train model
    callback = LearningRateScheduler(scheduler)
    model.fit(train_x, train_y, batch_size=32, epochs=100,
              validation_data=(test_x, test_y), callbacks=[callback])
    # Evaluate model
    scores = model.evaluate(test_x, test_y)
    return model, scores


def full_test():
    datasets = ('left_240x20', 'right_240x20', 'left_240x40', 'right_240x40',
                'left_480x80', 'right_480x80')
    # datasets = ('left_240x40', 'right_240x40', 'left_480x80', 'right_480x80')
    results_folder = Path.cwd() / 'mlp_results'
    results_folder.mkdir(exist_ok=True)
    results_file = results_folder / 'full_results.csv'
    if results_file.exists():
        results = pd.read_csv(results_file, index_col=0)
    else:
        results = pd.DataFrame(columns=['database', 'dataset', 'partition',
                                        'method', 'accuracy'])
    for dataset, part in product(datasets, range(1, 11)):
        train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
            dataset, part, mask_value=0, scale_dataset=True)
        orig_shape = (dataset.split('_')[-1]).split('x')
        orig_shape = (int(orig_shape[1]), int(orig_shape[0]))
        for i in range(6):
            method = "mlp_" + str(i)
            out_folder = "mlp_models"
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
            model, scores = train_mlp(train_x, train_y, test_x, test_y, i)
            print("Accuracy: %.2f%%" % (scores[1]*100))
            # Save model
            # model.save(
            #    out_folder / (dataset + '_' + str(part) + '_m' + str(i) + '.h5'))
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


if __name__ == '__main__':
    full_test()
