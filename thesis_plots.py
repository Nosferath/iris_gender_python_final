from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import MALES_LABEL, FEMALES_LABEL

ACC_LABEL = 'Accuracy'


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
                         dataset_name: str, max_y: int = None,
                         language='spanish'):
    """Generates a histogram describing the percentage of mask present
    in every image of the provided data.
    """
    texts = {
        'spanish': {
            'title': 'Máscaras en dataset',
            'correction': ' tras corrección',
            'xlabel': '% de máscara en la imagen',
            'ylabel': 'No. de imágenes'
        },
        'english': {
            'title': 'Masks in dataset',
            'correction': ' after correction',
            'xlabel': '% of mask in the image',
            'ylabel': 'N. of images'
        }
    }[language]

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
        title = f'{texts["title"]} {dataset_name}'
        out_name = dataset_name
        if dataset_name.endswith('_fixed'):
            title = title[:-6] + texts['correction']
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
        plt.xlabel(texts['xlabel'])
        plt.ylabel(texts['ylabel'])
        plt.tight_layout()
        # Save figure
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_folder / out_name)
        plt.close()


def _generate_mask_hists_by_gender(
    data_y, data_m, out_folder: str, dataset_name: str, use_pairs: bool,
    max_y: int = None, language: str = 'spanish'
):
    """Generates a histogram describing the percentage of mask present
    in every image of the provided data, separated by gender.
    """
    texts = {
        'spanish': {
            'f': 'Mujeres',
            'm': 'Hombres',
            'title': 'Máscaras por género en dataset',
            'corrected_pairs': '\ncorregidas, tras parear',
            'pairs': '\ntras parear',
            'corrected': '\ntras corrección',
            'xlabel': '% de máscara en la imagen',
            'ylabel': 'No. de imágenes'
        },
        'english': {
            'f': 'Females',
            'm': 'Males',
            'title': 'Masks per gender in dataset',
            'corrected_pairs': '\nafter correction and pairing',
            'pairs': '\nafter pairing',
            'corrected': '\nafter correction',
            'xlabel': '% of mask in the iris image',
            'ylabel': 'N. of images'
        }
    }[language]
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
        hist_f = sns.histplot(masks_f, bins=bins, color='red', hatch='//',
                              label=texts['f'], alpha=0.5)
        hist_m = sns.histplot(masks_m, bins=bins, color='blue', hatch=r'\\',
                              label=texts['m'], alpha=0.5)
        # Generate and set title
        title = f'{texts["title"]} GFI'
        out_name = 'g' + dataset_name
        if use_pairs and dataset_name.endswith('_fixed'):
            title = title + texts['corrected_pairs']
            out_name += '_pairs'
        elif use_pairs:
            title += texts['pairs']
            out_name += '_pairs'
        elif dataset_name.endswith('_fixed'):
            title = title + texts['corrected']
        else:
            title = '\n' + title
        out_name += '.png'
        plt.title(title)
        plt.legend()
        # Generate and set legend
        if max_y is not None:
            plt.ylim([0, max_y])
        plt.xlim([0, 80])
        plt.xlabel(texts['xlabel'])
        plt.ylabel(texts['ylabel'])
        plt.tight_layout()
        # plt.show()
        # Save figure
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_folder / out_name)  # , bbox_inches='tight')
        plt.close()
        plt.clf()


def _plot_pairs_analysis(thresholds, avg_scores, n_bad_pairs, out_folder,
                         dataset_name, language: str = 'spanish'):
    # avg_good_scores
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
    texts = {
        'spanish': {
            'growth': 'Crec. promedio',
            'n_high_growth': 'Pares c/alto crec.',
            'xlabel': 'Umbral [%]',
            'ylabel1': 'Crecimiento promedio [%]',
            'ylabel2': 'Pares c/alto crecimiento',
            'title': 'Análisis de pares, dataset',
            'fixed': ' (correg.)'
        },
        'english': {
            'growth': 'Avg. growth',
            'n_high_growth': 'Pairs w/high growth',
            'xlabel': 'Threshold [%]',
            'ylabel1': 'Average growth [%]',
            'ylabel2': 'Pairs w/high growth',
            'title': 'Pairs analysis, dataset',
            'fixed': ' (fixed)'
        }
    }[language]
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # display_name = dataset_name.replace('_fixed', texts['fixed'])
    display_name = 'GFI'
    if '_fixed' in dataset_name:
        display_name = f'{display_name}{texts["fixed"]}'
    df = pd.DataFrame({
        texts['xlabel']: thresholds,
        # 'avg_good_score': np.array(avg_good_scores) * 100,
        texts['growth']: np.array(avg_scores) * 100,
        texts['n_high_growth']: n_bad_pairs
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
            x=texts['xlabel'], y=texts['growth'], color=colors[0], style='-',
            legend=False
        )
        ax1.set_ylabel(texts['ylabel1'])
        ax1.yaxis.label.set_color(colors[0])
        ax2 = plt.twinx()
        df.plot(
            x=texts['xlabel'], y=texts['n_high_growth'], color=colors[1],
            style='--', legend=False, ax=ax2
        )
        ax2.set_ylabel(texts['ylabel2'])
        ax2.yaxis.label.set_color(colors[1])
        plt.title(f'{texts["title"]} {display_name}')
        ax1.figure.legend(loc='lower right', markerscale=0.5,
                          fontsize='xx-small')
        ax1.set_yticks(calculate_ticks(ax1, 5, 0.1))
        ax2.set_yticks(calculate_ticks(ax2, 5, 1))
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset_name}_thresh_analysis.png')
        plt.clf()


def _plot_pairs_histogram(hist, bins, out_folder, dataset, threshold,
                          max_y=None, n_bad_bins=1, language: str = 'spanish'):
    """Plots a distribution of the growth coefficient of pairs."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    texts = {
        'spanish': {
            'ylabel': '% de pares',
            'xlabel': '% de crecimiento',
            'title': 'Distrib. crecimiento pares,',
            'fixed': '(correg.)'
        },
        'english': {
            'ylabel': '% of pairs',
            'xlabel': '% of growth',
            'title': 'Pair growth distribution,',
            'fixed': '(fixed)'
        }
    }[language]

    if isinstance(hist, list):
        hist = np.array(hist)

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    display_name = 'GFI'
    if 'fixed' in dataset:
        display_name = f'{display_name} {texts["fixed"]}'

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
        plt.ylabel(texts['ylabel'])
        plt.xlabel(texts['xlabel'])
        if max_y is not None:
            if max_y < max(norm_hist):
                max_y = max(norm_hist) + 1
            plt.ylim([0, max_y])
        plt.title(f'{texts["title"]} {display_name}')
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset}_{threshold*100:.1f}.png')
        plt.clf()


def generate_pairs_plots(use_fixed_masks: bool, max_y=500, max_y_gender=260,
                         max_y_hist=25,
                         out_folder='../thesis_plots/masks_and_pairs',
                         language='spanish'):
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
    # Non-paired
    _generate_mask_hists(data_y, data_m, out_folder, dataset_name, max_y,
                         language)
    _generate_mask_hists_by_gender(data_y, data_m, out_folder, dataset_name,
                                   False, max_y_gender, language)
    # Paired
    spp_mat = calculate_spp_matrix(data_m[data_y == FEMALES_LABEL, :],
                                   data_m[data_y == MALES_LABEL, :])
    pairs, pair_scores = generate_pairs(data_y, data_m, spp_mat=spp_mat)
    paired_x, paired_m = apply_pairs(pairs, data_x, data_m, return_masks=True)
    _generate_mask_hists_by_gender(data_y, paired_m, out_folder, dataset_name,
                                   True, max_y_gender, language)
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
    # sorted_scores = np.sort(np.array(list(scores.values())))
    # np.savetxt(Path(out_folder) / 'scores.csv', sorted_scores[:, ::-1],
    #            delimiter=',')

    avg_scores = [a['avg_score'] for a in analysis.values()]
    n_bad_pairs = [a['n_bad_pairs'] for a in analysis.values()]
    _plot_pairs_analysis([t*100 for t in thresholds], avg_scores, n_bad_pairs,
                         out_folder, dataset_name, language)

    for t in thresholds:
        bad_bins = len(analysis[t]['histogram_full']) \
            - len(analysis[t]['histogram']) + 1
        _plot_pairs_histogram(np.array(analysis[t]['histogram_full']),
                              analysis[t]['bins_full'],
                              out_folder + '/histograms',
                              dataset_name,
                              threshold=t,
                              max_y=max_y_hist,
                              n_bad_bins=bad_bins,
                              language=language)


def generate_threshold_plots(out_folder='../thesis_plots/threshold_plots',
                             language='spanish'):
    from results_processing import process_pairs_thresh_results_to_df, \
        plot_pairs_thresh_results, anova_test
    texts = {
        'spanish': {
            'name_a': 'Umbral penaliz.',
            'name_b': 'Correcc. másc.',
            'title': 'Resultados de VGG+LSVM usando pares'
                     ' y distintos umbrales de penaliz.'
        },
        'english': {
            'name_a': 'Penaliz. thresh.',
            'name_b': 'Correct. masks',
            'title': 'VGG+LSVM results using pairs and'
                     ' different penalizing thresh.'
        }
    }[language]
    results_folder = Path('experiments/vgg_lsvm_pairs_thresh')
    df = process_pairs_thresh_results_to_df(results_folder)
    df_anova = anova_test(df, out_folder + '/anova', crit_a='threshold',
                          crit_b='fixed', name_a=texts['name_a'],
                          name_b=texts['name_b'], boxplot_title=None,
                          add_fixed_column=True)
    df_anova.to_csv(out_folder + '/anova/df_anova.csv')
    _ = plot_pairs_thresh_results(
        results_folder, out_folder + '/thresh_results.png',
        texts['title'],
        # '',
        use_thesis_labels=language == 'spanish'
    )


def generate_removebad_plots(language='spanish'):
    from results_processing import process_removebad_results
    texts = {
        'spanish': {
            'resolution': 'Resolución',
            'threshold': 'Umbral elim.',
            'test': 'Prueba'
        },
        'english': {
            'resolution': 'Resolution',
            'threshold': 'Removal thresh.',
            'test': 'Test'
        }
    }[language]
    df = process_removebad_results()
    print(df[(df.removed_bins == 1) & (df.database == 'gfi') &
             (df.test == 'vgg_full') & (df.dataset == '240x20')])
    df[texts['resolution']] = df['dataset']
    df[f'{ACC_LABEL} [%]'] = df['accuracy'] * 100
    df[texts['threshold']] = df['removed_bins'].apply(lambda x: f'{11 - x}%')
    df[texts['test']] = df['test'].apply(lambda x: x.capitalize())
    df = df.sort_values('removed_bins')
    for d in ('gfi', 'ndiris'):
        print(f'\nResultados database {d}')
        grouped = df[df.database == d].groupby(
            [texts['resolution'], texts['test'], texts['threshold']])
        description = grouped[f'{ACC_LABEL} [%]'].describe()
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
    for test in ('vgg_lsvm', 'vgg_full'):
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


def perform_one_way_anova_fixed():
    from results_processing import process_results_to_df, one_way_anova
    out_folder = '../thesis_plots/one_way_fixed/'
    for test in ('vgg_lsvm', 'vgg_full'):
        df = process_results_to_df(f'final_results/{test}/gfi_norm')
        for dataset in ('240x20', '240x40'):
            cur_df = df[df.dataset.isin([dataset, f'{dataset}_fixed'])]
            print(f'ANOVA with test {test} and dataset {dataset}')
            one_way_anova(cur_df, f'{out_folder}{test}_{dataset}', 'fixed',
                          'Use fixed masks', add_fixed_column=True)


def perform_two_way_anova_fixed_pairs():
    from results_processing import process_results_to_df, anova_test
    out_folder = '../thesis_plots/two_way_fixed_pairs'
    for test in ('vgg_lsvm', 'vgg_full'):
        non_pair_df = process_results_to_df(f'final_results/{test}/gfi_norm')
        non_pair_df['use_pairs'] = False
        pair_df = process_results_to_df(f'final_results/{test}/gfi_norm_pairs')
        pair_df['use_pairs'] = True
        df = pd.concat([non_pair_df, pair_df], ignore_index=True)
        for dataset in ('240x20', '240x40'):
            cur_df = df[df.dataset.isin([dataset, f'{dataset}_fixed'])]
            print(f'ANOVA with test {test} and dataset {dataset}')
            anova_test(cur_df, f'{out_folder}{test}_{dataset}', 'use_pairs',
                       'fixed', 'Use mask pairs', 'Use fixed masks',
                       add_fixed_column=True)


def perform_one_way_anova_ndiris_pairs():
    from results_processing import process_results_to_df, one_way_anova
    out_folder = '../thesis_plots/one_way_ndiris_pairs/'
    pd.set_option('display.float_format',
                  lambda x: np.format_float_scientific(x))
    for test in ('vgg_lsvm', 'vgg_full'):
        non_pair_df = process_results_to_df(f'final_results/{test}/ndiris')
        non_pair_df['use_pairs'] = False
        pair_df = process_results_to_df(f'final_results/{test}/ndiris_pairs')
        pair_df['use_pairs'] = True
        df = pd.concat([non_pair_df, pair_df], ignore_index=True)
        for dataset in ('240x20', '240x40'):
            cur_df = df[df.dataset == dataset]
            print(f'ANOVA with test {test} and dataset {dataset}')
            one_way_anova(cur_df, f'{out_folder}{test}_{dataset}', 'use_pairs',
                          'Use mask pairs')


def perform_one_way_anova_gfi_periocular():
    from results_processing import process_results_to_df, one_way_anova
    out_folder = '../thesis_plots/one_way_gfi_peri/'
    for test in ('vgg_lsvm', 'vgg_full'):
        df_peri = process_results_to_df(f'final_results/{test}/gfi_peri')
        df_peri['image_type'] = 'periocular'
        df_norm = process_results_to_df(f'final_results/{test}/gfi_norm')
        df_norm['image_type'] = 'normalized'
        df_norm = df_norm[df_norm.dataset == '240x40']
        cur_df = pd.concat([df_peri, df_norm])
        print(f'ANOVA test with test {test}')
        one_way_anova(cur_df, f'{out_folder}{test}', 'image_type',
                      'Image type')


def comparison_barplot():
    data_normalized = [
        ['Thomas et al., 2007', 0.8],
        ['Lagree and Bowyer, 2011', 0.62],
        ['Bansal et al., 2012', 0.8306],
        ['Tapia et al., 2015', 0.9133],
        ['Fairhurst et al., 2015', 0.8974],
        ['Tapia et al., 2016', 0.91],
        ['Bobeldyk and Ross, 2016', 0.69],
        ['Tapia and Aravena, 2017', 0.8466],
        ['Kuehlkamp et al., 2017', 0.66],
        ['Tapia and Perez, 2019', 0.9545],
        ['Kuehlkamp and Bowyer, 2019', 0.6340],
        ['Tapia and Arellano, 2019', 0.9466],
        ['Eskandari and Sharifi, 2019', 0.6667],
    ]
    data_normalized = [
        {'title': a, 'accuracy': b*100, 'type': 'normalized'}
        for a, b in data_normalized
    ]
    data_periocular = [
        ['Bobeldyk and Ross, 2016', 0.857],
        ['Singh et al., 2017', 0.8317],
        ['Tapia and Viedma, 2017', 0.8969],
        ['Rattani et al., 2017', 0.9020],
        ['Kuelkamp et al., 2017', 0.80],
        ['Rattani et al., 2018', 0.9000],
        ['Tapia and Aravena, 2018', 0.8726],
        ['Viedma and Tapia, 2018', 0.8593],
        ['Bobeldyk and Ross, 2018', 0.8590],
        ['Bobeldyk and Ross, 2019', 0.8540],
        ['Kuehlkamp and Bowyer, 2019', 0.8080],
        ['Manyala et al., 2019', 0.9463],
        ['Viedma et al., 2019', 0.8689],
        ['Tapia et al., 2019', 0.9190],
        ['Tapia and Arellano, 2019 (2)', 0.9190],
        ['Tapia et al., 2019 (2)', 0.9015],
        ['Alonso-Fernandez et al., 2020', 0.9200],
        ['Raja et al., 2020', 0.8159],
        ['Alonso-Fernandez et al., 2021', 0.7510],
        ['Vetrekar et al., 2021', 0.7572],
    ]
    data_periocular = [
        {'title': a, 'accuracy': b*100, 'type': 'periocular'}
        for a, b in data_periocular
    ]
    data = [*data_normalized, *data_periocular]
    df = pd.DataFrame(data)
    df['year'] = df.loc[:, 'title'].apply(
        lambda x: x.split(', ')[1].split(' ')[0]
    )
    df = df.sort_values(['year', 'title'])
    print(df)
    sns.set(rc={'figure.figsize': (16, 7)})
    sns.set_palette('muted')
    sns.set_style('ticks')
    sns.set_context('talk')
    plot: plt.Axes = sns.barplot(data=df, x='title', y='accuracy', hue='type')
    plot.set_xlabel('')
    plot.set_ylabel('Accuracy [%]')
    plot.set_ylim([50, 100])
    plot.set_title(
        'Comparison of gender classification results using normalized '
        'and periocular iris images'
    )
    plot.grid(True)
    fig = plt.gcf()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig('/home/nosferath/TESIS/Defensa Tesis/state-of-the-art.png')
    plt.show()
    return df, plot, fig


def show_mask_correction_differences(bottom_adjust=0.3, max_y=500,
                                     out_folder='../Defensa Tesis/'):
    """Plots a histogram of the masks distribution after fix, and an
    outline of the distribution before the fix.
    """
    from load_data import load_dataset_both_eyes
    from load_data_utils import generate_partitions_both_eyes, \
        balance_partition

    # Load data
    dataset_name = '240x20'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    _, data_y, data_m, _ = balance_partition(*train_data)
    original_average = np.mean(data_m)
    print(f'Promedio original: {original_average:.2%}')
    dataset_name += '_fixed'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    _, data_y_fixed, data_m_fixed, _ = balance_partition(*train_data)
    fixed_average = np.mean(data_m_fixed)
    print(f'Promedio fixed: {fixed_average:.2%}. Diferencia: '
          f'{original_average - fixed_average:.2%}')
    before_fix = np.sum(data_m * 100 / (240 * 20), axis=1)
    after_fix = np.sum(data_m_fixed * 100 / (240 * 20), axis=1)

    # Set the start and end values for binning
    start = 0
    end = 80
    bin_width = 5

    # Calculate histogram values for "beforeFix" with bin width of 5
    num_bins_before = int((end - start) / bin_width)
    hist_before, bins_before = np.histogram(before_fix, bins=num_bins_before,
                                            range=(start, end))
    hist_heights_before = np.insert(hist_before, 0, 0)
    bin_edges_before = np.arange(start, end + bin_width, bin_width)

    # Calculate histogram values for "afterFix" with bin width of 5
    # num_bins_after = int((end - start) / bin_width)
    # hist_after, bins_after = np.histogram(after_fix, bins=num_bins_after,
    #                                       range=(start, end))
    # hist_heights_after = np.insert(hist_after, 0, 0)
    bin_edges_after = np.arange(start, end + bin_width, bin_width)

    # Plot data
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.6, 4.8)}):
        sns.histplot(after_fix, kde=False, bins=bin_edges_after,
                     label='Después de corrección', linewidth=1)
        plt.step(bin_edges_before, hist_heights_before, color='red',
                 linewidth=2, linestyle='-', label='Antes de corrección')
        # plt.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center',
        #            ncol=2, fancybox=False, shadow=False)
        plt.legend([], frameon=False)
        if max_y is not None:
            plt.ylim([0, max_y])
        plt.xlim([0, 80])
        plt.xlabel('% de máscara en la imagen')
        plt.ylabel('No. de imágenes')
        plt.title('Máscaras en dataset 240x20 tras corrección')

        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)
        # Save figure
        out_folder = Path(out_folder)
        out_name = 'before_after_fix.png'
        plt.savefig(out_folder / out_name, bbox_inches='tight',
                    transparent=False)
        plt.close()


def show_mask_pairs_differences(bottom_adjust=0.3, max_y=500,
                                out_folder='../Defensa Tesis/'):
    """Plots a histogram of the masks distribution after fix, and an
    outline of the distribution before the fix.
    """
    from load_data import load_dataset_both_eyes
    from load_data_utils import generate_partitions_both_eyes, \
        balance_partition
    from mask_pairs import calculate_spp_matrix, generate_pairs, apply_pairs

    # Load data
    dataset_name = '240x20'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    data_x, data_y, data_m, _ = balance_partition(*train_data)
    original_average = np.mean(data_m)
    print(f'Promedio original: {original_average:.2%}')
    spp_mat = calculate_spp_matrix(data_m[data_y == FEMALES_LABEL, :],
                                   data_m[data_y == MALES_LABEL, :])
    pairs, pair_scores = generate_pairs(data_y, data_m, spp_mat=spp_mat)
    paired_x, paired_m = apply_pairs(pairs, data_x, data_m, return_masks=True)
    paired_average = np.mean(paired_m)
    print(f'Promedio paired: {paired_average:.2%}. Diferencia:'
          f' {original_average - paired_average:.2%}')
    before_pairs = np.sum(data_m * 100 / (240 * 20), axis=1)
    after_pairs = np.sum(paired_m * 100 / (240 * 20), axis=1)

    # Set the start and end values for binning
    start = 0
    end = 80
    bin_width = 5

    # Calculate histogram values for "beforeFix" with bin width of 5
    num_bins_before = int((end - start) / bin_width)
    hist_before, bins_before = np.histogram(before_pairs, bins=num_bins_before,
                                            range=(start, end))
    hist_heights_before = np.insert(hist_before, 0, 0)
    bin_edges_before = np.arange(start, end + bin_width, bin_width)

    # Calculate histogram values for "afterFix" with bin width of 5
    # num_bins_after = int((end - start) / bin_width)
    # hist_after, bins_after = np.histogram(after_fix, bins=num_bins_after,
    #                                       range=(start, end))
    # hist_heights_after = np.insert(hist_after, 0, 0)
    bin_edges_after = np.arange(start, end + bin_width, bin_width)

    # Plot data
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.6, 4.8)}):
        sns.histplot(after_pairs, kde=False, bins=bin_edges_after,
                     label='Después de parear', linewidth=1)
        plt.step(bin_edges_before, hist_heights_before, color='red',
                 linewidth=2, linestyle='-', label='Antes de parear')
        # plt.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center',
        #            ncol=2, fancybox=False, shadow=False)
        plt.legend([], frameon=False)
        if max_y is not None:
            plt.ylim([0, max_y])
        plt.xlim([0, 80])
        plt.xlabel('% de máscara en la imagen')
        plt.ylabel('No. de imágenes')
        plt.title('Máscaras en dataset 240x20 tras pareo')

        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)
        # Save figure
        out_folder = Path(out_folder)
        out_name = 'before_after_pairs.png'
        plt.savefig(out_folder / out_name, bbox_inches='tight',
                    transparent=False)
        plt.close()


def show_mask_both_differences(bottom_adjust=0.3, max_y=500,
                               out_folder='../Defensa Tesis/'):
    """Plots a histogram of the masks distribution after fix, and an
    outline of the distribution before the fix.
    """
    from load_data import load_dataset_both_eyes
    from load_data_utils import generate_partitions_both_eyes, \
        balance_partition
    from mask_pairs import calculate_spp_matrix, generate_pairs, apply_pairs

    # Load data
    dataset_name = '240x20'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    _, data_y, data_m, _ = balance_partition(*train_data)
    original_average = np.mean(data_m)
    print(f'Promedio original: {original_average:.2%}')
    dataset_name += '_fixed'
    data, males, females = load_dataset_both_eyes(dataset_name, True, False)
    train_data, test_data, _ = generate_partitions_both_eyes(data, males,
                                                             females, 1, 0.2)
    data_x_fixed, data_y_fixed, data_m_fixed, _ = \
        balance_partition(*train_data)
    fixed_average = np.mean(data_m_fixed)
    print(f'Promedio fixed: {fixed_average:.2%}. Diferencia: '
          f'{original_average - fixed_average:.2%}')
    spp_mat = calculate_spp_matrix(
        data_m_fixed[data_y_fixed == FEMALES_LABEL, :],
        data_m_fixed[data_y_fixed == MALES_LABEL, :]
    )
    pairs, pair_scores = generate_pairs(data_y_fixed, data_m_fixed,
                                        spp_mat=spp_mat)
    paired_x, paired_m = apply_pairs(pairs, data_x_fixed, data_m_fixed,
                                     return_masks=True)
    fixed_paired_average = np.mean(paired_m)
    print(f'Promedio fixed+paired: {fixed_paired_average:.2%}. Diferencia:'
          f' {original_average - fixed_paired_average:.2%}')
    before_pairs = np.sum(data_m * 100 / (240 * 20), axis=1)
    after_pairs = np.sum(paired_m * 100 / (240 * 20), axis=1)

    # Set the start and end values for binning
    start = 0
    end = 80
    bin_width = 5

    # Calculate histogram values for "beforeFix" with bin width of 5
    num_bins_before = int((end - start) / bin_width)
    hist_before, bins_before = np.histogram(before_pairs, bins=num_bins_before,
                                            range=(start, end))
    hist_heights_before = np.insert(hist_before, 0, 0)
    bin_edges_before = np.arange(start, end + bin_width, bin_width)

    # Calculate histogram values for "afterFix" with bin width of 5
    # num_bins_after = int((end - start) / bin_width)
    # hist_after, bins_after = np.histogram(after_fix, bins=num_bins_after,
    #                                       range=(start, end))
    # hist_heights_after = np.insert(hist_after, 0, 0)
    bin_edges_after = np.arange(start, end + bin_width, bin_width)

    # Plot data
    with sns.plotting_context('talk'), \
            sns.axes_style(rc={'grid.color': '#b0b0b0',
                               'axes.grid': True,
                               'axes.axisbelow': True,
                               'grid.linestyle': '-',
                               'figure.figsize': (6.6, 4.8)}):
        sns.histplot(after_pairs, kde=False, bins=bin_edges_after,
                     label='Después de C+P', linewidth=1)
        plt.step(bin_edges_before, hist_heights_before, color='red',
                 linewidth=2, linestyle='-', label='Antes de C+P')
        # plt.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center',
        #            ncol=2, fancybox=False, shadow=False)
        plt.legend([], frameon=False)
        if max_y is not None:
            plt.ylim([0, max_y])
        plt.xlim([0, 80])
        plt.xlabel('% de máscara en la imagen')
        plt.ylabel('No. de imágenes')
        plt.title('Máscaras en dataset 240x20 tras corr. y pareo')

        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)
        # Save figure
        out_folder = Path(out_folder)
        out_name = 'before_after_both.png'
        plt.savefig(out_folder / out_name, bbox_inches='tight',
                    transparent=False)
        plt.close()


def rotate_and_shift_images(periocular_path, normalized_path, delta=20,
                            pause=5, center_x=291, center_y=239,
                            output_path='../Defensa Tesis/gif/rotation.gif',
                            save_as_frames=True):
    # Thanks ChatGPT!
    from PIL import Image
    # Load the images
    periocular_image = Image.open(periocular_path)
    normalized_image = Image.open(normalized_path)

    # Convert the images to numpy arrays
    # periocular_array = np.array(periocular_image)
    normalized_array = np.array(normalized_image)

    # Determine the maximum width and height among the periocular and
    # normalized images
    max_width = max(periocular_image.width, normalized_image.width)
    max_height = max(periocular_image.height, normalized_image.height)

    # Define the rotation angle and horizontal shift parameters
    rotation_angles = np.concatenate(
        (np.zeros(pause), np.linspace(0, -360, 36))
    )  # Rotate in 10-degree increments
    horizontal_shifts = np.concatenate(
        (np.zeros(pause),
         np.linspace(0, normalized_array.shape[1],
                     len(rotation_angles) - pause))
    )

    # Create a blank canvas for the frame with a width and height based
    # on the maximum dimensions
    frame_width = max_width + normalized_image.width + delta
    frame_height = max_height  # max(max_width, max_height)
    frame = Image.new("RGB", (frame_width, frame_height))

    # Calculate Y position for the normalized image
    normalized_y_position = frame_height // 2 - normalized_array.shape[0] // 2

    # Create a list to store the frames for the GIF
    frames = []

    # Iterate over the rotation angles and horizontal shifts
    for angle, shift in zip(rotation_angles, horizontal_shifts):
        # Rotate the periocular image
        rotated_periocular = \
            periocular_image.rotate(angle, center=(center_x, center_y))

        # Calculate the offset to align the rotated periocular and
        # normalized images
        periocular_offset = (frame_height - rotated_periocular.height) // 2

        # Paste the rotated periocular image onto the frame
        frame.paste(rotated_periocular, (0, periocular_offset))

        # Calculate the shifted position for the normalized image
        normalized_shifted_position = int(shift) % normalized_image.width

        # Perform the horizontal shift using array slicing and concatenation
        shifted_array = np.concatenate(
            (normalized_array[:, normalized_shifted_position:],
             normalized_array[:, :normalized_shifted_position]),
            axis=1
        )

        # Create a new image from the shifted array
        shifted_normalized = Image.fromarray(shifted_array)

        # Paste the shifted normalized image to the right of the rotated
        # periocular image
        frame.paste(shifted_normalized,
                    (rotated_periocular.width + delta, normalized_y_position))

        # Append the frame to the list of frames
        frames.append(frame.copy())

    # Save the frames as a GIF file
    if save_as_frames:
        output_path = Path(output_path)
        output_name = output_path.stem
        frame: Image
        for i, frame in enumerate(frames):
            frame.save(
                output_path.parent / f'{output_name}-{i}.png',
                format='png'
            )
        print(f'Frames saved to {output_path.parent}')
    else:
        frames[0].save(
            output_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=200,
            loop=0,
        )

        print(f"GIF saved to {output_path}")
