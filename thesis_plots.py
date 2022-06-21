from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _calculate_mask_percentages(data_y, data_m, genders: bool):
    """Calculates the percentage of mask for each sample of the dataset.
    If genders is True, these values are separated by gender."""
    # Process masks into percentages
    n_feats = data_m.shape[1]
    mask_percentages = np.sum(data_m, axis=1)
    mask_percentages = 100 * mask_percentages / n_feats
    if genders:
        masks_percentage_female = mask_percentages[data_y == 0]  # Female
        masks_percentage_male = mask_percentages[data_y == 1]  # Male
        return masks_percentage_female, masks_percentage_male
    return mask_percentages

# def _plot_mask_distributions():


def _generate_mask_hists(data_y: np.array, data_m: np.array, out_folder: str,
                         dataset_name: str, max_y: int = None):
    """Generates a histogram describing the percentage of mask present
    in every image of the provided data.
    """
    # Obtain mask percentages
    masks = _calculate_mask_percentages(data_y, data_m, genders=False)
    # Define histogram bins
    bins = np.linspace(0, 100, 21)
    # Generate histogram
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'figure.figsize': (6.4, 4.8)}):
        fig, ax = plt.subplots()
        hist = sns.histplot(masks, bins=bins, ax=ax)
        title = f'Máscaras en dataset {dataset_name}'
        out_name = dataset_name
        if dataset_name.endswith('_fixed'):
            title = title[:-6] + ' tras corrección'
        # if use_pairs:
        #     title += '\ntras emparejar'
        #     out_name += '_pairs'
        # else:
        #     title = '\n' + title
        out_name += '.png'
        ax.set_title(title)
        if max_y is not None:
            ax.set_ylim([0, max_y])
        plt.xlim([0, 80])
        plt.xlabel('% de máscara en la imagen')
        plt.ylabel('No. de imágenes')
        plt.tight_layout()
        # Save figure
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_folder / out_name)
        plt.close()


def _generate_mask_hists_by_gender(
    data_y, data_m, out_folder: str, dataset_name: str, use_pairs: bool,
    max_y: int = None
):
    """Generates a histogram describing the percentage of mask present
    in every image of the provided data, separated by gender.
    """
    # Obtain mask percentages
    masks_f, masks_m = _calculate_mask_percentages(data_y, data_m,
                                                   genders=True)
    # Define histogram bins
    bins = np.linspace(0, 100, 21)
    # Generate histogram
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'figure.figsize': (6.4, 4.8)}):
        hist_f = sns.histplot(masks_f, bins=bins, color='red',
                              label='Mujeres', alpha=0.5)
        hist_m = sns.histplot(masks_m, bins=bins, color='blue',
                              label='Hombres', alpha=0.5)
        # Generate and set title
        title = f'Máscaras por género en dataset {dataset_name}'
        out_name = 'g' + dataset_name
        if use_pairs and dataset_name.endswith('_fixed'):
            title = title[:-6] + '\ncorregidas, tras emparejar'
            out_name += '_pairs'
        elif use_pairs:
            title += '\ntras emparejar'
            out_name += '_pairs'
        elif dataset_name.endswith('_fixed'):
            title = title[:-6] + '\ntras corrección'
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
        plt.savefig(out_folder / out_name)  # , bbox_inches='tight')
        plt.close()
        plt.clf()


def generate_pairs_plots(use_fixed_masks: bool, max_y=500, max_y_gender=260):
    """Generates plots about the pairs: histogram of mask distribution
    per gender before and after pairs, growth scores and number of bad
    pairs at different thresholds.
    """
    from load_data import load_dataset_both_eyes
    from load_data_utils import generate_partitions_both_eyes, \
        balance_partition
    from mask_pairs import generate_pairs, apply_pairs

    dataset_name = '240x20'
    if use_fixed_masks:
        dataset_name += '_fixed'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    data_x, data_y, data_m, _ = balance_partition(*train_data)
    out_folder = '../thesis_plots/masks_and_pairs'
    # Non-paired
    _generate_mask_hists(data_y, data_m, out_folder, dataset_name, max_y)
    _generate_mask_hists_by_gender(data_y, data_m, out_folder, dataset_name,
                                   False, max_y_gender)
    # Paired
    pairs, pair_scores = generate_pairs(data_y, data_m)
    paired_x, paired_m = apply_pairs(pairs, data_x, data_m, return_masks=True)
    _generate_mask_hists_by_gender(data_y, paired_m, out_folder, dataset_name,
                                   True, max_y_gender)
