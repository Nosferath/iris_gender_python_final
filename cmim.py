from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from constants import CMIM_FOLDER, CMIM_STD_FOLDER


def load_cmim_array_from_path(filepath, fix_indexing=True):
    """Loads the CMIM array and fixes 1-indexing."""
    cmim_array = loadmat(str(filepath))
    cmim_array = cmim_array['cmimArray']
    # Fix 1-indexing
    if cmim_array.min() == 1 and fix_indexing:
        cmim_array = cmim_array - 1
    # Fix 2D to 1D
    if len(cmim_array.shape) == 2:
        cmim_array = cmim_array.reshape(-1)
    return cmim_array


def load_cmim_array(dataset_name: str, pair_method: str):
    """Loads the CMIM array and fixes 1-indexing. If pair_method is None
    or False, the array is loaded from the CMIM_FOLDER. Otherwise, it is
    loaded from the CMIM_STD_FOLDER regardless of the pair method.
    """
    path = Path(CMIM_STD_FOLDER) if pair_method else Path(CMIM_FOLDER)
    path = path / f'{dataset_name}.mat'
    return load_cmim_array_from_path(path)


def generate_cmim_visualization(cmim_array: np.ndarray,
                                base_image: np.ndarray,
                                n_parts_total: int,
                                n_parts_displ: int):
    """Generates a visualization of the CMIM array by showing a sort of
    heatmap of the most important features. Up to 8 colors can be
    currently displayed.
    """
    from matplotlib import cm
    assert n_parts_displ <= n_parts_total, "Features to display must be " \
                                           "less than or equal to total."
    assert 2 <= len(base_image.shape) <= 3, "Base image must be 2D or 3D."
    assert cmim_array.size <= np.product(base_image.shape[:2])
    # Define colors
    colors = cm.get_cmap('Paired').colors
    colors = [[v*255 for v in c] for c in colors]
    colors = colors[5::-1] + colors[-2:]
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
    pairs : str or None/False
        Set to None/False if pairs are not to be used. Otherwise, set to the
        pairing method name.
    n_parts_total : int
        Number of parts in which to divide the total number of features.
    n_parts_displ : int
        Out of the total, number of parts that will actually be
        displayed.
    out_folder : str
        Path to where the visualizations will be saved.
    """
    from results_processing import generate_mask_visualization
    cmim_folder = Path(cmim_folder)
    cmim_files = cmim_folder.glob('*.mat')
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in cmim_files:
        dataset_name = file.stem
        cmim_array = load_cmim_array_from_path(file)
        base_image = generate_mask_visualization(dataset_name, pairs)
        visualization = generate_cmim_visualization(cmim_array,
                                                    base_image,
                                                    n_parts_total,
                                                    n_parts_displ)
        img = Image.fromarray(visualization)
        img.save(str(out_folder / f"{dataset_name}.png"))


def visualize_mask_prevalence(cmim_folder: str, pairs: str, out_folder: str,
                              avg_width: int, n_parts_total: int, partition=1):
    """Visualizes the percentage of masked examples for each feature, in
    the same order as they were selected. The values are individual,
    i.e., at each number of features, only the percentage for that
    feature is shown.

    Parameters
    ----------
    cmim_folder : str
        Folder with CMIM arrays.
    pairs : str or None/False
        Set to None/False if pairs are not to be used. Otherwise, set to the
        pairing method name.
    out_folder : str
        Path to where the visualizations will be saved.
    avg_width : str
        Width of the moving average window
    n_parts_total : int
        Number of parts in which to divide the total number of features.
        Used for visualizing which features are in which group with
        vertical colored areas.
    partition : int
        Partition to use for splitting the dataset. Both train and test
        are being used currently, so this has no effect. Default: 1.
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from load_partitions import load_partitions_pairs
    # Define colors
    colors = cm.get_cmap('Paired').colors
    colors = [[v for v in c] for c in colors]
    colors = colors[5::-1] + colors[-2:]
    # Define folders
    cmim_folder = Path(cmim_folder)
    cmim_files = cmim_folder.glob('*.mat')
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for file in cmim_files:
        dataset_name = file.stem
        cmim_array = load_cmim_array_from_path(file)
        _, _, train_m, _, _, _, _, _ = load_partitions_pairs(
            dataset_name, partition, mask_value=0, scale_dataset=True,
            pair_method=pairs
        )
        masks = train_m
        # Calculate sum of masks, in cmim-order
        sum_masks = masks.sum(axis=0)[cmim_array]
        # Generate number of features
        sum_feats = np.repeat(masks.shape[0], len(cmim_array))
        # Calculate prevalence for each feature
        preval = np.divide(sum_masks, sum_feats)
        x = np.arange(len(preval)) + 1
        # Calculate moving average
        moving = np.convolve(preval, np.ones(avg_width), 'same') / avg_width
        # Generate plot
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots()
        ax.plot(x, 100*preval, '.', label='Mask prevalence')
        ax.plot(x[:-int(avg_width/2)], 100*moving[:-int(avg_width/2)],
                linewidth=2.5, label=f'Moving avg. (w={avg_width})')
        areas = np.linspace(1, x.max(), n_parts_total+1)
        # for i in range(n_parts_total):
        #    ax.axvspan(areas[i], areas[i+1], facecolor=colors[i], alpha=0.5)
        ax.grid('on')
        ax.legend()
        ax.set_xlabel('Feature order')
        ax.set_ylabel('Mask prevalence [%]')
        # ax.set_xlim([0, len(preval)+1])
        ax.set_xlim([0, 50])
        # ax.set_ylim([0, 100*preval.max()])
        ax.set_ylim([0, 100 * preval[:50].max()+1])
        ax.set_title(f'Mask prevalence per feature, dataset={dataset_name}')
        fig.savefig(str(out_folder / f'{dataset_name}_zoom.png'))
        fig.clf()
        plt.close(fig)
        del sum_masks, sum_feats, preval
