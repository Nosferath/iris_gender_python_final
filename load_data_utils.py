from pathlib import Path
from typing import List

import numpy as np

DEFAULT_ROOT_PATH = Path('S:/NUND_fixed_masks/')


def get_labels_df(eye: str, root_path=DEFAULT_ROOT_PATH):
    """Loads the labels for each image from the GFI.txt

    Filenames are kept without their extension.
    """
    import pandas as pd

    filename = f'List_{eye.lower()}_GFI.txt'
    df = pd.read_csv(root_path / filename, sep='\t', header=None)
    df.columns = ['filename', 'gender']
    df.loc[:, 'filename'] = df.filename.apply(lambda x: x.split('.')[0])
    return df


def fix_labels_df(images: List[str], classes: List[int], eye: str,
                  root_path=DEFAULT_ROOT_PATH):
    """Adds the missing labels to the GFI.txt.
    DOES NOT CHECK FOR DUPLICATES.

    Parameters
    ----------
    images : List of strings
        Contains the STEMS of the images' filenames
    classes : List of ints
        Contains the classes of the images, as 0s or 1s
    eye : str
        Eye of the dataset. Either 'left' or 'right'
    root_path : Pathlib Path, optional
        Path where the GFI.txt file is located.
    """
    with open(root_path / f'List_{eye.lower()}_GFI.txt', 'a') as f:
        for img, c in zip(images, classes):
            print(f'Adding file {img} with label {c} to labels')
            f.write(f'\n{img}.tiff\t{c}')


def save_raw_dataset(data: np.ndarray, labels: np.ndarray, masks: np.ndarray,
                     image_paths: List, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out_file,
        data=data,
        labels=labels,
        masks=masks,
        image_paths=np.array(image_paths)
    )
