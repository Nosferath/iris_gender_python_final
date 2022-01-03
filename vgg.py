import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16


def load_vgg_model():
    model = VGG16(weights='imagenet')
    print(model.summary())
    return model


def load_vgg_model_features():
    """Loads the VGG-16 model up to the first FC layer, for
    feature extraction.
    """
    from tensorflow.keras.models import Model
    base_model = VGG16(weights='imagenet')
    layer = 'fc1'
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model


def prepare_data_for_vgg(train_x: np.ndarray, train_y: np.ndarray,
                         test_x: np.ndarray, test_y: np.ndarray):
    from skimage.color import gray2rgb
    from skimage.transform import resize
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from utils import find_shape

    orig_shape = find_shape(n_features=train_x.shape[1])
    train_x = gray2rgb(train_x.reshape((-1, *orig_shape)))
    test_x = gray2rgb(test_x.reshape((-1, *orig_shape)))
    out_train_x = np.zeros((train_x.shape[0], 224, 224, 3))
    for i in range(train_x.shape[0]):
        cur_img = train_x[i]
        cur_img = resize(cur_img, output_shape=(224, 224), mode='edge',
                         order=3)
        out_train_x[i, :, :, :] = cur_img
    out_test_x = np.zeros((test_x.shape[0], 224, 224, 3))
    for i in range(test_x.shape[0]):
        cur_img = test_x[i]
        cur_img = resize(cur_img, output_shape=(224, 224), mode='edge',
                         order=3)
        out_test_x[i, :, :, :] = cur_img

    out_train_x = preprocess_input(out_train_x)
    out_test_x = preprocess_input(out_test_x)

    return out_train_x, train_y, out_test_x, test_y
