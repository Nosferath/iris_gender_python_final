from typing import Tuple

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


def load_vgg_model_finetune(lr=0.01):
    """Loads the VGG-16 model without the original FC layers,
    """
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.optimizers import Adam
    base_model = VGG16(include_top=False, weights='imagenet',
                       input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Freeze original layers
    for layer in base_model.layers:
        layer.trainable = False

    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def load_vgg_model_features():
    """Loads the VGG-16 model up to the first FC layer, for
    feature extraction.
    """
    base_model = VGG16(weights='imagenet')
    layer = 'fc1'
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model


def _prepare_data_for_vgg(data_x: np.ndarray, orig_shape: Tuple[int, int]):
    from skimage.color import gray2rgb
    from skimage.transform import resize
    from tensorflow.keras.applications.vgg16 import preprocess_input

    data_x = gray2rgb(data_x.reshape((-1, *orig_shape)))
    out_data_x = np.zeros((data_x.shape[0], 224, 224, 3))
    for i in range(data_x.shape[0]):
        cur_img = data_x[i]
        cur_img = resize(cur_img, output_shape=(224, 224), mode='edge',
                         order=3)
        out_data_x[i, :, :, :] = cur_img

    out_data_x = preprocess_input(out_data_x)

    return out_data_x


def prepare_data_for_vgg(data_x: np.ndarray):
    from utils import find_shape

    orig_shape = find_shape(n_features=data_x.shape[1])
    return _prepare_data_for_vgg(data_x, orig_shape)


def labels_to_onehot(labels):
    """Convert binary labels to one-hot encoding"""
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(labels.reshape((-1, 1))).astype('float32')


def prepare_periocular_for_vgg(data_x: np.ndarray):
    from constants import PERIOCULAR_SHAPE
    return _prepare_data_for_vgg(data_x, PERIOCULAR_SHAPE)
