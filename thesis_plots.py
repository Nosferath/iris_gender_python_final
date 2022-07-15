from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import MALES_LABEL, FEMALES_LABEL


def _calculate_mask_percentages(data_y, data_m, genders: bool):
    """Calculates the percentage of mask for each sample of the dataset.
    If genders is True, these values are separated by gender."""
    # Process masks into percentages
    n_feats = data_m.shape[1]
    mask_percentages = np.sum(data_m, axis=1)
    mask_percentages = 100 * mask_percentages / n_feats
    if genders:
        masks_percentage_female = mask_percentages[data_y == FEMALES_LABEL]
        masks_percentage_male = mask_percentages[data_y == MALES_LABEL]
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
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.4, 4.8)}):
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
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.4, 4.8)}):
        hist_f = sns.histplot(masks_f, bins=bins, color='red',
                              label='Mujeres', alpha=0.5)
        hist_m = sns.histplot(masks_m, bins=bins, color='blue',
                              label='Hombres', alpha=0.5)
        # Generate and set title
        title = f'Máscaras por género en dataset {dataset_name}'
        out_name = 'g' + dataset_name
        if use_pairs and dataset_name.endswith('_fixed'):
            title = title[:-6] + '\ncorregidas, tras parear'
            out_name += '_pairs'
        elif use_pairs:
            title += '\ntras parear'
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


def _plot_pairs_analysis(thresholds, avg_scores, n_bad_pairs, out_folder,
                         dataset_name):  # avg_good_scores
    """Plots average growth scores and number of bad pairs for multiple
    thesholds.

    Parameters
    ----------
    thresholds : iterable
        List of the penalization thresholds for each score and number
        of bad pairs
    avg_scores : iterable
        List of the growth scores obtained for each threshold
    n_bad_pairs : iterable
        List of the number of bad pairs for each threshold
    out_folder : string or pathlib.Path
        Folder for storing the resulting plot
    dataset_name : string
        Name of the dataset, only used for the plot title and filename
    """
    from utils import calculate_ticks
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        'Umbral [%]': thresholds,
        # 'avg_good_score': np.array(avg_good_scores) * 100,
        'Crec. promedio': np.array(avg_scores) * 100,
        'Pares c/alto crec.': n_bad_pairs
    })
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.4, 4.8)}):
        colors = sns.color_palette()[:2]
        ax1 = df.plot(
            # x='threshold', y='avg_good_score', color=colors[0], legend=False
            x='Umbral [%]', y='Crec. promedio', color=colors[0], legend=False
        )
        ax1.set_ylabel('Crecimiento promedio [%]')
        ax1.yaxis.label.set_color(colors[0])
        ax2 = plt.twinx()
        df.plot(
            x='Umbral [%]', y='Pares c/alto crec.', color=colors[1],
            legend=False, ax=ax2
        )
        ax2.set_ylabel('Pares c/alto crecimiento')
        ax2.yaxis.label.set_color(colors[1])
        plt.title(f'Análisis de pares, dataset {dataset_name}')
        ax1.figure.legend(loc='lower right', markerscale=0.5,
                          fontsize='x-small')
        ax1.set_yticks(calculate_ticks(ax1, 5, 0.1))
        ax2.set_yticks(calculate_ticks(ax2, 5, 1))
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset_name}_thresh_analysis.png')
        plt.clf()


def _plot_pairs_histogram(hist, bins, out_folder, dataset, threshold,
                          max_y=None, n_bad_bins=1):
    """Plots a distribution of the growth coefficient of pairs."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(hist, list):
        hist = np.array(hist)

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    with sns.plotting_context('talk'):
        colors = sns.color_palette()[:2]

        norm_hist = 100 * hist / np.sum(hist)
        x = np.array(bins[:-1])
        x = 100*(x + (x[0] + x[1]) / 2)
        delta = (x[0] + x[1]) / 2
        ticks = x - delta/2
        # labels = [f'{v:.1f}' for v in ticks]
        labels = [f'{v:.0f}' for v in ticks]
        bar_colors = [colors[0]] * (len(norm_hist) - n_bad_bins)
        bar_colors.extend([colors[1]] * n_bad_bins)
        ax = plt.bar(x, norm_hist, width=delta * 0.9, color=bar_colors,
                     edgecolor='black', linewidth=2)
        # Generate labels for small values
        label_hist = list(norm_hist)
        for i in range(len(label_hist)):
            if label_hist[i] > 0.5 or label_hist[i] == 0:
                label_hist[i] = ""
            else:
                label_hist[i] = f"{label_hist[i]:.1f}"[1:]
        plt.bar_label(ax, label_hist)
        # plt.bar_label(ax, hist)
        plt.xticks(ticks[::2], labels[::2])
        plt.grid(True, axis='y')
        plt.ylabel('% de pares')
        plt.xlabel('% de crecimiento')
        if max_y is not None:
            if max_y < max(norm_hist):
                max_y = max(norm_hist) + 1
            plt.ylim([0, max_y])
        plt.title(f'Distrib. crecimiento pares, {dataset}')
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset}_{threshold*100:.1f}.png')
        plt.clf()


def generate_pairs_plots(use_fixed_masks: bool, max_y=500, max_y_gender=260,
                         max_y_hist=25):
    """Generates plots about the pairs: histogram of mask distribution
    per gender before and after pairs, growth scores and number of bad
    pairs at different thresholds.
    """
    from load_data import load_dataset_both_eyes
    from load_data_utils import generate_partitions_both_eyes, \
        balance_partition
    from mask_pairs import calculate_spp_matrix, generate_pairs, apply_pairs
    from results_processing import analyze_pairs

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
    spp_mat = calculate_spp_matrix(data_m[data_y == FEMALES_LABEL, :],
                                   data_m[data_y == MALES_LABEL, :])
    pairs, pair_scores = generate_pairs(data_y, data_m, spp_mat=spp_mat)
    paired_x, paired_m = apply_pairs(pairs, data_x, data_m, return_masks=True)
    _generate_mask_hists_by_gender(data_y, paired_m, out_folder, dataset_name,
                                   True, max_y_gender)
    # Pairs per threshold
    thresholds = list(np.arange(0, 0.20, 0.005))
    scores = {}
    for t in thresholds:
        if t == 0.1:
            scores[t] = pair_scores
            continue
        cur_pairs, cur_scores = generate_pairs(data_y, data_m, threshold=t,
                                               spp_mat=spp_mat)
        scores[t] = cur_scores
    analysis = {t: analyze_pairs(scores[t]) for t in thresholds}

    avg_scores = [a['avg_score'] for a in analysis.values()]
    n_bad_pairs = [a['n_bad_pairs'] for a in analysis.values()]
    _plot_pairs_analysis([t*100 for t in thresholds], avg_scores, n_bad_pairs,
                         out_folder, dataset_name)

    for t in thresholds:
        bad_bins = len(analysis[t]['histogram_full']) \
            - len(analysis[t]['histogram']) + 1
        _plot_pairs_histogram(np.array(analysis[t]['histogram_full']),
                              analysis[t]['bins_full'],
                              out_folder + '/histograms',
                              dataset_name,
                              threshold=t,
                              max_y=max_y_hist,
                              n_bad_bins=bad_bins)


def generate_threshold_plots():
    from results_processing import process_pairs_thresh_results_to_df, \
        plot_pairs_thresh_results, anova_test
    results_folder = Path('experiments/vgg_lsvm_pairs_thresh')
    out_folder = '../thesis_plots/threshold_plots'
    df = process_pairs_thresh_results_to_df(results_folder)
    df_anova = anova_test(df, out_folder + '/anova', crit_a='threshold',
                          crit_b='fixed', name_a='Umbral penaliz.',
                          name_b='Correcc. Másc.', boxplot_title=None,
                          add_fixed_column=True)
    df_anova.to_csv(out_folder + '/anova/df_anova.csv')
    _ = plot_pairs_thresh_results(
        results_folder, out_folder + '/thresh_results.png',
        # 'Resultados de VGG+LSVM usando emparej. y distintos umbrales',
        '',
        use_thesis_labels=True
    )


def generate_removebad_plots():
    from results_processing import process_removebad_results
    df = process_removebad_results()
    df['Resolución'] = df['dataset']
    df['Exactitud [%]'] = df['accuracy'] * 100
    df['Umbral elim.'] = df['removed_bins'].apply(lambda x: f'{11 - x}%')
    df['Prueba'] = df['test'].apply(lambda x: x.capitalize())
    df = df.sort_values('removed_bins')
    for d in ('gfi', 'ndiris'):
        print(f'\nResultados database {d}')
        grouped = df[df.database == d].groupby(
            ['Resolución', 'Prueba', 'Umbral elim.'])
        description = grouped['Exactitud [%]'].describe()
        print(description[['mean', 'std']].applymap(lambda x: f'{x:0.2f}'))


def perform_one_way_anova_removebad():
    from itertools import product
    from results_processing import process_removebad_results, one_way_anova
    out_folder = '../thesis_plots/one_way_removebad/'
    df = process_removebad_results()
    database = 'gfi'
    for test, dataset in product(['vgg_lsvm', 'vgg_full'],
                                 ['240x20', '240x40']):
        cur_df = df[(df.database == database) &
                    (df.test == test) &
                    (df.dataset == dataset) &
                    (df.removed_bins.apply(lambda x: 1 <= x <= 4))]
        print(f'ANOVA with test {test} and dataset {dataset}')
        one_way_anova(cur_df, f'{out_folder}{test}_{dataset}', 'removed_bins',
                      'Removed bins')


def perform_one_way_anova_pairs():
    from results_processing import process_results_to_df, one_way_anova
    out_folder = '../thesis_plots/one_way_pairs/'
    for test in ('vgg_full', 'vgg_lsvm'):
        non_pair_df = process_results_to_df(f'final_results/{test}/gfi_norm')
        non_pair_df['use_pairs'] = False
        pair_df = process_results_to_df(f'final_results/{test}/gfi_norm_pairs')
        pair_df['use_pairs'] = True
        df = pd.concat([non_pair_df, pair_df], ignore_index=True)
        for dataset in ('240x20', '240x40'):
            cur_df = df[df.dataset == dataset]
            print(f'ANOVA with test {test} and dataset {dataset}')
            one_way_anova(cur_df, f'{out_folder}{test}_{dataset}', 'use_pairs',
                          'Use mask pairs')
