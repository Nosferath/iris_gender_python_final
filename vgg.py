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


def prepare_data_for_vgg(data_x: np.ndarray):
    from skimage.color import gray2rgb
    from skimage.transform import resize
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from utils import find_shape

    orig_shape = find_shape(n_features=data_x.shape[1])
    data_x = gray2rgb(data_x.reshape((-1, *orig_shape)))
    out_data_x = np.zeros((data_x.shape[0], 224, 224, 3))
    for i in range(data_x.shape[0]):
        cur_img = data_x[i]
        cur_img = resize(cur_img, output_shape=(224, 224), mode='edge',
                         order=3)
        out_data_x[i, :, :, :] = cur_img

    out_data_x = preprocess_input(out_data_x)

    return out_data_x
