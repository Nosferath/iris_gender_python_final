from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import PAIR_METHOD
from load_partitions import load_partitions_cmim


def calculate_mask_percentage(dataset_name: str, use_pairs: bool,
                              genders: bool):
    """Calculates the percentage of mask for each sample of the dataset.
    If genders is True, these values are separated by gender."""
    # Load dataset and merge mask arrays
    pair_method = PAIR_METHOD if use_pairs else False
    _, train_y, train_m, _, _, test_y, test_m, _ = load_partitions_cmim(
        dataset_name=dataset_name,
        partition=1,  # Not relevant
        mask_value=0,  # Not relevant
        scale_dataset=True,  # Not relevant
        pair_method=pair_method,
        n_cmim=0
    )
    if not genders:
        y_arr = np.hstack([train_y, test_y])
        masks = np.vstack([train_m, test_m])
    else:
        # Merging train and test does not make sense if using
        # genders to show pairs/non-pairs difference
        y_arr = train_y
        masks = train_m
    # Process masks into percentages
    n_feats = masks.shape[1]
    masks = np.sum(masks, axis=1)
    masks = 100 * masks / n_feats
    if genders:
        masks_0 = masks[y_arr == 0]  # Female
        masks_1 = masks[y_arr == 1]  # Male
        return masks_0, masks_1
    return masks


def generate_mask_hists(dataset_name: str, use_pairs: bool, max_y: int = None):
    """Generates a histogram describing the percentage of mask present
    in every image of the dataset.
    """
    # Obtain mask percentages
    masks = calculate_mask_percentage(dataset_name, use_pairs, genders=False)
    # Define histogram bins
    bins = np.linspace(0, 100, 11)
    # Generate histogram
    sns.set_context("talk")
    fig, ax = plt.subplots()
    hist = sns.histplot(masks, bins=bins, ax=ax)
    title = f'Máscaras en dataset {dataset_name}'
    if use_pairs:
        title += '\ntras emparejar'
    else:
        title = '\n' + title
    ax.set_title(title)
    if max_y is not None:
        ax.set_ylim([0, max_y])
    plt.tight_layout()
    plt.show()


def generate_mask_hists_by_gender(dataset_name: str, use_pairs: bool,
                                  max_y: int = None):
    """Generates a histogram describing the percentage of mask present
    in every image of the dataset, separated by gender.
    """
    # Obtain mask percentages
    masks_f, masks_m = calculate_mask_percentage(dataset_name, use_pairs,
                                                 genders=True)
    # Define histogram bins
    bins = np.linspace(0, 100, 11)
    # Generate histogram
    sns.set_theme()
    sns.set_context("paper")
    fig, axs = plt.subplots(2, 1)
    hist_f = sns.histplot(masks_f, bins=bins, ax=axs[0], color='red',
                          label='Mujeres')
    hist_m = sns.histplot(masks_m, bins=bins, ax=axs[1], color='blue',
                          label='Hombres')
    # Generate and set title
    title = f'Máscaras por género en dataset {dataset_name}'
    if use_pairs:
        title += ' tras emparejar'
    fig.suptitle(title)
    # Generate and set legend
    # handles_labels = [ax.get_legend_handles_labels() for ax in axs]
    # handles, labels = [sum(ll, []) for ll in zip(*handles_labels)]
    # leg = plt.legend(handles, labels)
    # leg.set_in_layout(True)
    if max_y is None:
        max_f = np.max(np.histogram(masks_f, bins)[0])
        max_m = np.max(np.histogram(masks_m, bins)[0])
        max_y = int(1.05 * max(max_m, max_f))
    for ax in axs:
        ax.set_ylim([0, max_y])
        ax.legend()
    plt.tight_layout()
    plt.show()


def generate_mask_hists_by_gender2(
        dataset_name: str, use_pairs: bool, max_y: int = None,
        out_folder: str = r'S:\Tesis\Figuras\Histogramas'):
    """Generates a histogram describing the percentage of mask present
    in every image of the dataset, separated by gender.
    """
    # Obtain mask percentages
    masks_f, masks_m = calculate_mask_percentage(dataset_name, use_pairs,
                                                 genders=True)
    # Define histogram bins
    bins = np.linspace(0, 100, 21)
    # Generate histogram
    sns.set_theme()
    sns.set_context("talk")
    hist_f = sns.histplot(masks_f, bins=bins, color='red',
                          label='Mujeres', alpha=0.5)
    hist_m = sns.histplot(masks_m, bins=bins, color='blue',
                          label='Hombres', alpha=0.5)
    # Generate and set title
    title = f'Máscaras por género en dataset {dataset_name}'
    out_name = dataset_name
    if use_pairs:
        title += '\ntras emparejar'
        out_name += '_pairs'
    else:
        title = '\n' + title
    out_name += '.png'
    plt.title(title)
    plt.legend()
    # Generate and set legend
    if max_y is not None:
        plt.ylim([0, max_y])
    plt.xlim([0, 80])
    plt.xlabel('% de máscara en la imagen')
    plt.ylabel('No. de imágenes')
    plt.tight_layout()
    # plt.show()
    # Save figure
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_folder / out_name, bbox_inches='tight')
    plt.close()
    plt.clf()


def generate_masked_iris_image(x_arr: np.ndarray, m_arr: np.ndarray):
    """Reshapes the x_arr to its final shape, and applies the mask in
    green for visualization."""
    shapes = {4800: (20, 240), 9600: (40, 240), 38400: (80, 480)}
    shape = shapes[x_arr.size]
    if x_arr.max() <= 1:
        x_arr = x_arr * 255
    x_arr = x_arr.reshape(shape).astype('uint8')
    m_arr = m_arr.reshape(shape)
    # Extend x to 3-D
    x_arr = np.tile(x_arr[..., np.newaxis], (1, 1, 3))
    x_arr[m_arr == 1] = np.array((0, 255, 0), dtype='uint8')
    return x_arr


def visualize_pairs(dataset_name: str, pair_idx: int = 0,
                    out_folder: str = r'S:\Tesis\Figuras\Pares'):
    """Visualizes iris before and after pairing."""
    from PIL import Image
    from pairs import load_pairs_array
    # Load dataset and pairs
    train_x, _, train_m, _, _, _, _, _ = load_partitions_cmim(
        dataset_name=dataset_name,
        partition=1,  # Not relevant
        mask_value=0,  # Not relevant
        scale_dataset=True,  # Not relevant
        pair_method=False,
        n_cmim=0
    )
    pairs = load_pairs_array(dataset_name, PAIR_METHOD, partition=1)
    # Select pair and extract
    cur_pair = pairs[pair_idx, :]
    x_a = train_x[int(cur_pair[0]), :]
    m_a = train_m[int(cur_pair[0]), :]
    x_b = train_x[int(cur_pair[1]), :]
    m_b = train_m[int(cur_pair[1]), :]
    # Generate visualizations
    x_a_unpaired = generate_masked_iris_image(x_a, m_a)
    x_b_unpaired = generate_masked_iris_image(x_b, m_b)
    m_pair = np.any(np.vstack([m_a, m_b]), axis=0)
    x_a_paired = generate_masked_iris_image(x_a, m_pair)
    x_b_paired = generate_masked_iris_image(x_b, m_pair)
    # Save visualizations
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    out_names = ['a_unpaired.png', 'b_unpaired.png',
                 'a_paired.png', 'b_paired.png']
    out_arrs = [x_a_unpaired, x_b_unpaired, x_a_paired, x_b_paired]
    for out_name, out_arr in zip(out_names, out_arrs):
        img = Image.fromarray(out_arr)
        img.save(out_folder / out_name)
