import pickle
from pathlib import Path

import numpy as np
from skimage.color import gray2rgb
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import preprocess_input

from constants import PERIOCULAR_SHAPE
from load_data import load_peri_dataset_both_eyes
from vgg_utils import labels_to_onehot, load_periocular_botheyes_pre_vgg, \
    load_vgg_model_features

# Peri images start as uint8
# After scaling, they become float64

def generate_data():
    """Used to generate the periocular _fix dataset. As opposed to the
    previous periocular data, this one does not go through unnecesary
    scaling and dtype changing. Additionally, we used bilinear interpol-
    ation instead of bicubic.
    """
    out_size = (224, 224, 3)
    all_data, males_set, females_set = load_peri_dataset_both_eyes(True)
    for eye in all_data:
        data_x = all_data[eye][0]
        if data_x.max() == 1:
            data_x = data_x * 255
        data_x = gray2rgb(data_x.reshape((-1, *PERIOCULAR_SHAPE)))
        out_data_x = np.zeros((data_x.shape[0], *out_size))
        for i in range(data_x.shape[0]):
            cur_img = data_x[i]
            if PERIOCULAR_SHAPE != out_size[:2]:
                cur_img = resize(
                    cur_img, output_shape=out_size[:2], mode='edge', order=1
                )
            out_data_x[i, :, :, :] = cur_img

        out_data_x = preprocess_input(out_data_x)
        all_data[eye][0] = out_data_x

        all_data[eye][1] = labels_to_onehot(all_data[eye][1])
        
    np.savez_compressed(
        'data_peri/both_pre_vgg_fix.npz',
        all_data=all_data,
        males_set=males_set,
        females_set=females_set
    )


def fix_vgg_feats_peri():
    """Fix the periocular data used for vgg+lsvm. The data is processed
    by the VGG to extract features, and the labels are returned to
    normal encoding instead of one-hot.
    """
    all_data, males_set, females_set = load_periocular_botheyes_pre_vgg('_fix')
    model = load_vgg_model_features()
    for eye in all_data:
        data_x = all_data[eye][0]
        all_data[eye][0] = model.predict(data_x)
        # Remove one-hot encoding
        data_y = all_data[eye][1]
        all_data[eye][1] = data_y.argmax(axis=1)
    
    np.savez_compressed(
        'data_peri/both_vgg.npz',
        all_data=all_data,
        males_set=males_set,
        females_set=females_set
    )



def review_fix_results(
    path='experiments/full_vgg/fixing_peri_1/both_peri_fix_1'
):
    path = Path(path) / 'callback_results.pickle'
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(results)
    return results
