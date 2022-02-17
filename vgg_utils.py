from typing import Tuple

import numpy as np


def set_precision_to_16_bits():
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


def get_vgg_fc_architecture(architecture, base_model_output):
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    fc_sizes = {'test1': 2048, 'test2': 1024, 'test3': 1024, 'test4': 1024,
                'peri1': 4096}
    fc_size = fc_sizes[architecture]
    if architecture == 'test1':
        x = Flatten()(base_model_output)
        x = Dense(fc_size, activation='relu')(x)
        x = Dense(fc_size, activation='relu')(x)  # New FC2
    elif architecture in ['test2', 'test3']:
        x = Flatten()(base_model_output)
        x = Dense(fc_size, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(int(fc_size / 2), activation='relu')(x)
        x = Dense(int(fc_size / 4), activation='relu')(x)
    elif architecture == 'test4':
        x = Flatten()(base_model_output)
        x = Dense(fc_size, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(int(fc_size / 2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(int(fc_size / 4), activation='relu')(x)
    elif architecture == 'peri1':
        x = Flatten()(base_model_output)
        x = Dense(fc_size, activation='relu')(x)
        x = Dense(int(fc_size / 2), activation='relu')(x)
        x = Dense(int(fc_size / 4), activation='relu')(x)
    else:
        raise ValueError(f'Unrecognized architecture option: {architecture}')
    predictions = Dense(2, activation='softmax')(x)
    return predictions


def load_vgg_model_finetune(lr=0.001, input_shape=(224, 224, 3),
                            architecture='test4'):
    """Loads the VGG-16 model without the original FC layers, freezing
    the original conv layers, and adding new FC layers for training on
    iris.
    """
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.metrics import categorical_accuracy
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    set_precision_to_16_bits()

    base_model = VGG16(include_top=False, weights='imagenet',
                       input_shape=input_shape)
    outputs = get_vgg_fc_architecture(architecture, base_model.output)
    model = Model(inputs=base_model.input, outputs=outputs)
    # Freeze original layers
    for layer in base_model.layers:
        layer.trainable = False

    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def load_vgg_model_features():
    """Loads the VGG-16 model up to the first FC layer, for
    feature extraction.
    """
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.models import Model
    set_precision_to_16_bits()

    base_model = VGG16(weights='imagenet')
    layer = 'fc1'
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model


def _prepare_data_for_vgg(data_x: np.array, orig_shape: Tuple[int, int],
                          scale_data=False, img_shapes=(224, 224, 3)):
    from skimage.color import gray2rgb
    from skimage.transform import resize
    from tensorflow.keras.applications.vgg16 import preprocess_input

    if data_x.max() == 1:
        data_x = data_x * 255
    if scale_data:
        from load_data_utils import scale_data_by_row
        data_x = (scale_data_by_row(data_x) * 255).astype('uint8')
    data_x = gray2rgb(data_x.reshape((-1, *orig_shape)))
    out_data_x = np.zeros((data_x.shape[0], *img_shapes))
    for i in range(data_x.shape[0]):
        cur_img = data_x[i]
        if orig_shape != img_shapes[:2]:
            cur_img = resize(
                cur_img, output_shape=img_shapes[:2], mode='edge', order=3
            )
        out_data_x[i, :, :, :] = cur_img

    out_data_x = preprocess_input(out_data_x)

    return out_data_x


def prepare_data_for_vgg(data_x: np.array, preserve_shape=False):
    """Prepares normalized iris images for use with VGG feature
    extractor.
    """
    from utils import find_shape

    orig_shape = find_shape(n_features=data_x.shape[1])
    if preserve_shape:
        img_shapes = (max(orig_shape[0], 32), orig_shape[1], 3)
    else:
        img_shapes = (224, 224, 3)
    return _prepare_data_for_vgg(data_x, orig_shape, scale_data=False,
                                 img_shapes=img_shapes)


def prepare_botheyes_for_vgg(all_data, preserve_shape=False):
    for eye in all_data:
        cur_x = all_data[eye][0]
        all_data[eye][0] = prepare_data_for_vgg(
            cur_x, preserve_shape=preserve_shape
        )

    return all_data


def labels_to_onehot(labels):
    """Convert binary labels to one-hot encoding"""
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(labels.reshape((-1, 1))).astype('float32')


def labels_to_onehot_botheyes(all_data):
    for eye in all_data:
        cur_y = all_data[eye][1]
        all_data[eye][1] = labels_to_onehot(cur_y)

    return all_data


def prepare_periocular_for_vgg(data_x: np.ndarray, scale_data=False):
    from constants import PERIOCULAR_SHAPE
    return _prepare_data_for_vgg(data_x, PERIOCULAR_SHAPE,
                                 scale_data=scale_data)


def prepare_periocular_botheyes_for_vgg(all_data, scale_data=False):
    from constants import PERIOCULAR_SHAPE
    for eye in all_data:
        cur_x = all_data[eye][0]
        all_data[eye][0] = _prepare_data_for_vgg(cur_x, PERIOCULAR_SHAPE,
                                                 scale_data=scale_data)
    return all_data


def load_periocular_vgg(eye: str):
    """Loads the periocular dataset that has already been processed by
    VGG to extract features. Used for the LSVM test.
    """
    from load_data import load_peri_dataset_from_npz
    return load_peri_dataset_from_npz(f'{eye}_vgg')


def load_periocular_pre_vgg(eye: str):
    """Data is already prepared and labels are one-hot-encoded. Used for
    the full VGG test.
    """
    from load_data import load_peri_dataset_from_npz
    return load_peri_dataset_from_npz(f'{eye}_pre_vgg')


def _load_periocular_botheyes_vgg(filename):
    from constants import ROOT_PERI_FOLDER
    loaded = np.load(ROOT_PERI_FOLDER + f'/{filename}.npz', allow_pickle=True)
    all_data = loaded['all_data'].item()
    males_set = loaded['males_set'].item()
    females_set = loaded['females_set'].item()
    return all_data, males_set, females_set


def load_periocular_botheyes_vgg():
    """Dataset already processed using VGG. For LSVM."""
    return _load_periocular_botheyes_vgg('both_vgg')


def load_periocular_botheyes_pre_vgg(subtype=""):
    return _load_periocular_botheyes_vgg(f'both_pre_vgg{subtype}')


def load_data(data_type: str, **kwargs):
    if data_type == 'periocular_vgg_feats':
        eye = kwargs['eye']
        data, labels, _ = load_periocular_vgg(eye)
        data = (data, labels)
    elif data_type == 'iris_vgg_feats':
        from load_data import load_iris_dataset
        feat_model = load_vgg_model_features()
        dataset_name = kwargs['dataset_name']
        data, labels = load_iris_dataset(dataset_name, None)
        data = prepare_data_for_vgg(data)
        data = feat_model.predict(data)
        data = (data, labels)
        del feat_model
    elif data_type == 'iris_botheyes_vgg_feats':
        from load_data import load_dataset_both_eyes
        dataset_name = kwargs['dataset_name']
        all_data, males_set, females_set = load_dataset_both_eyes(dataset_name)
        feat_model = load_vgg_model_features()
        all_data = prepare_botheyes_for_vgg(all_data)
        for eye in all_data:
            cur_x = all_data[eye][0]
            all_data[eye][0] = feat_model.predict(cur_x)
        data = (all_data, males_set, females_set)
        del feat_model
    elif data_type == 'periocular_botheyes_vgg_feats':
        all_data, males_set, females_set = load_periocular_botheyes_vgg()
        data = (all_data, males_set, females_set)
    else:
        raise ValueError(f'Option {data_type} not recognized')
    return data
