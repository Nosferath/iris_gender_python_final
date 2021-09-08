from pathlib import Path

from matplotlib import cm
import numpy as np
from PIL import Image
from scipy.io import loadmat

from utils import generate_mask_visualization


def load_cmim_array(filepath):
    cmim_array = loadmat(str(filepath))
    cmim_array = cmim_array['cmimArray']
    return cmim_array


def generate_cmim_visualization(cmim_array: np.ndarray,
                                base_image: np.ndarray,
                                n_parts_total: int,
                                n_parts_displ: int):
    """Generates a visualization of the CMIM array by showing a sort of
    heatmap of the most important features.
    """
    assert n_parts_displ <= n_parts_total, "Features to display must be " \
                                           "less than or equal to total."
    assert 2 <= len(base_image.shape) <= 3, "Base image must be 2D or 3D."
    assert cmim_array.size == np.product(base_image.shape[:2])
    # Fix 1-indexing
    if cmim_array.min() == 1:
        cmim_array = cmim_array - 1
    # Define colors
    colors = cm.get_cmap('Paired').colors
    colors = [[v*255 for v in c] for c in colors]
    if len(colors) < n_parts_displ:
        print('[WARN] Feature classes to display is more than available '
              'colors. Setting to number of colors instead.')
        n_parts_displ = len(colors)
    # Define conversion from index to coords
    h, w = base_image.shape[:2]

    def index_to_coords(idx):
        return np.unravel_index(idx, (h, w))

    # Define conversion from feature importance to color
    n_color = int(len(cmim_array) / n_parts_total)  # Features per color

    def index_to_color(imp):
        return colors[imp // n_color]

    # Convert base_image to 3D if necessary
    if len(base_image.shape) != 3 or base_image.shape[2] != 3:
        out_image = np.stack([base_image] * 3, axis=2)
    else:
        out_image = base_image.copy()

    # Apply colors to out_image
    for i in range(n_parts_displ * n_color):
        cur_cmim = cmim_array[i]
        y, x = index_to_coords(cur_cmim)
        cur_color = index_to_color(i)
        out_image[y, x, :] = cur_color

    return out_image


def review_all_cmim(cmim_folder: str, pairs: str, n_parts_total: int,
                    n_parts_displ: int, out_folder: str):
    """Generates visualizations for all CMIM arrays in the folder.

    Parameters
    ----------
    cmim_folder : str
        Folder with CMIM arrays.
    pairs : str or None
        Set to none if pairs are not to be used. Otherwise, set to the
        pairing method name.
    n_parts_total : int
        Number of parts in which to divide the total number of features.
    n_parts_displ : int
        Out of the total, number of parts that will actually be
        displayed.
    out_folder : str
        Path to where the visualizations will be saved.
    """
    cmim_folder = Path(cmim_folder)
    cmim_files = cmim_folder.glob('*.mat')
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in cmim_files:
        dataset_name = file.stem
        cmim_array = load_cmim_array(file)
        base_image = generate_mask_visualization(dataset_name, pairs)
        visualization = generate_cmim_visualization(cmim_array,
                                                    base_image,
                                                    n_parts_total,
                                                    n_parts_displ)
        img = Image.fromarray(visualization)
        img.save(str(out_folder / f"{dataset_name}.png"))
