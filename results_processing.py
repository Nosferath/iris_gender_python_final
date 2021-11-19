from pathlib import Path
import pickle

import numpy as np

from constants import datasets, MODEL_PARAMS_FOLDER
from utils import grid_plot


def generate_cv_grid_plot(dataset_name: str, cv: int, params_folder: str):
    """Generates visualizations of the cross-validation results for the
    chosen dataset.
    """
    assert cv in (1, 2), 'cv must be 1 or 2'
    cv_file = Path(params_folder) / f'cv{cv}_{dataset_name}.pickle'
    if not cv_file.exists():
        raise FileNotFoundError(f'{params_folder}/{cv_file.name} not found')
    with open(cv_file, 'rb') as f:
        cv_results: dict = pickle.load(f)
    params = cv_results['params']
    results = cv_results['mean_test_score']
    results_std = cv_results['std_test_score']
    # Get unique values of params
    param_types = list(params[0].keys())
    name_a = param_types[0]
    name_b = param_types[1]
    list_a = np.unique(np.array([p[name_a] for p in params]))
    list_b = np.unique(np.array([p[name_b] for p in params]))
    # Convert results to a grid
    n_a = len(list_a)
    n_b = len(list_b)
    results_grid = np.zeros((n_a, n_b), dtype='float64')
    std_grid = np.zeros((n_a, n_b), dtype='float64')
    for a in range(n_a):
        for b in range(n_b):
            i = b + n_b*a
            results_grid[a, b] = results[i]
            std_grid[a, b] = results_std[i]
    # Generate results plot
    fig, ax = grid_plot(list_a, list_b, results_grid * 100)
    ax.set_ylabel(name_a)
    ax.set_xlabel(name_b)
    ax.set_title(f'{dataset_name}, CV{cv} results')
    # Generate results std plot
    fig_std, ax_std = grid_plot(list_a, list_b, std_grid * 100)
    ax_std.set_ylabel(name_a)
    ax_std.set_xlabel(name_b)
    ax_std.set_title(f'{dataset_name}, CV{cv} results, std')
    return fig, fig_std


def review_cv_results(params_folder: str, out_folder: str):
    import matplotlib.pyplot as plt
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        for cv in (1, 2):
            try:
                fig, fig_std = generate_cv_grid_plot(
                    dataset, cv, params_folder)
            except FileNotFoundError:
                print(f'File for {dataset} CV{cv} not found. Skipping.')
                continue
            figname = f'{dataset}_cv{cv}.png'
            figstdname = f'{dataset}_cv{cv}_std.png'
            fig.savefig(str(out_folder / figname), bbox_inches='tight',
                        transparent=False)
            fig_std.savefig(str(out_folder / figstdname), bbox_inches='tight',
                            transparent=False)
            plt.close('all')


def review_results(results_folder: str, print_train=False):
    results_folder = Path(results_folder)
    if not results_folder.exists():
        raise ValueError(f'{results_folder} not found')
    for file in results_folder.glob('*.pickle'):
        with open(file, 'rb') as f:
            cur_results = pickle.load(f)
        if 'acc_train' in cur_results[0] and print_train:
            train_results = np.array([d['acc_train'] for d in cur_results])
            mean = train_results.mean() * 100
            std = train_results.std() * 100
            print(f'Train {file.stem}:\t{mean:.2f} ± {std:.2f}')
        cur_results = np.array([d['accuracy'] for d in cur_results])
        mean = cur_results.mean() * 100
        std = cur_results.std() * 100
        print(f'{file.stem}:\t{mean:.2f} ± {std:.2f}')


def review_params(params_folder: str, verbose: int):
    """Verbosity 1: Print CV results. Verbosity 2+: Print all CV info"""
    params_folder = Path(MODEL_PARAMS_FOLDER) / params_folder
    keys = [f'split{i}_test_score' for i in range(5)]
    keys.extend(['mean_test_score', 'std_test_score', 'rank_test_score'])
    for file in params_folder.glob('*.pickle'):
        if file.name.startswith('cv') and not verbose:
            continue
        with open(file, 'rb') as f:
            cur_params = pickle.load(f)
        if file.name.startswith('cv'):
            if verbose == 1:
                cur_params = {k: cur_params[k] for k in keys}
        print(f'{file.stem}:\t{cur_params}')


def visualize_all_masks(out_folder: str, use_pairs: bool):
    """Generates visualizations for all masks"""
    import matplotlib.pyplot as plt
    from constants import PAIR_METHOD
    from load_partitions import generate_mask_visualization
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    pairs = PAIR_METHOD if use_pairs else False
    for dataset in datasets:
        masks = generate_mask_visualization(dataset, pairs)
        plt.imshow(masks, cmap='jet_r')
        plt.axis('off')
        out_path = out_folder / f'{dataset}.png'
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def anova_test(results_folder: str, std_results_folder: str, out_folder: str,
               crit_a='fixed', crit_b='uses_pairs', name_a='Fixed masks',
               name_b='Mask pairs', boxplot_title=None):
    """Performs a two-way ANOVA test on the results obtained for this
     classifier. The variables to compare are pairing and mask-fixing.
     """
    from itertools import product
    from bioinfokit.analys import stat
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    name_a = crit_a if name_a is None else name_a
    name_b = crit_b if name_b is None else name_b
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    results_folder = Path(results_folder)
    std_results_folder = Path(std_results_folder)
    results_files = list(results_folder.glob('*.pickle'))
    results_files += list(std_results_folder.glob('*.pickle'))
    # Generate dataframe with results
    df = pd.DataFrame(columns=[
        'dataset', 'fixed', 'uses_pairs', 'partition', 'result'
    ])
    for file in results_files:
        dataset = file.stem
        fixed = dataset.endswith('fixed')
        if fixed:
            dataset = dataset[:-6]
        uses_pairs = 'std' in file.parent.name
        with open(file, 'rb') as f:
            results = np.array([r['accuracy'] for r in pickle.load(f)])
        for i in range(results.size):
            df = df.append({
                'dataset': dataset, 'fixed': fixed, 'uses_pairs': uses_pairs,
                'partition': i + 1, 'result': results[i]
            }, ignore_index=True)
    # Test whether samples follow normal distribution with Q-Q plot
    crit_a_values = df[crit_a].unique()
    crit_b_values = df[crit_b].unique()
    for c_a, c_b in product(crit_a_values, crit_b_values):
        condition = (df[crit_a] == c_a) & (df[crit_b] == c_b)
        fig, ax = plt.subplots()
        stats.probplot(df[condition]['result'], dist='norm', plot=ax)
        ax.set_title(f"Probability plot - {name_a}: {c_a}, {name_b}: {c_b}",
                     fontsize=15)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(labelsize=15)
        fig.tight_layout()
        fig.savefig(out_folder / f"qq_a{str(c_a)}_b{str(c_b)}.png")
        plt.close()
    # Test whether samples follow normal distribution with Shapiro-Wilk
    res = stat()
    model_str = f'result ~ C({crit_a}) + C({crit_b}) + C({crit_a}):C({crit_b})'
    res.tukey_hsd(
        df=df, res_var='result', xfac_var=[crit_a, crit_b],
        anova_model=model_str
    )
    _, pvalue = stats.shapiro(res.anova_model_out.resid)
    if pvalue <= 0.01:
        print(f'Shapiro-Wilk test p-value is less or equal than 0.01 '
              f'({pvalue:.2f})\nCondition is NOT satisfied.')
    else:
        print(f'Shapiro-Wilk test p-value is greater than 0.01 ({pvalue:.2f})'
              f'\nCondition is satisfied.')
    # Plot distribution of data using histograms
    for c_a in crit_a_values:
        cur_df = df[df[crit_a] == c_a]
        # min_val = np.floor(cur_df.result.min()*100)/100
        # max_val = np.ceil(cur_df.result.max()*100)/100
        min_val = 0.49
        max_val = 0.67
        # sns.set_style("whitegrid")
        sns.set_context("talk")
        ax = sns.histplot(data=cur_df, x='result', hue=crit_b, binwidth=0.01,
                          binrange=(min_val, max_val))
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.set_title(f'Distribution of results - {name_a}={c_a}',
                     fontsize=17)
        ax.set_xlabel('Accuracy')
        ax.set_ylim((0, 20))
        ax.set_xlim((0.49, 0.68))
        ax.set_xticks(np.arange(0.49, 0.67, 0.03))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(labelsize=15)
        plt.tight_layout()
        legend = ax.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax.legend(handles, crit_b_values, title=name_b)
        plt.savefig(out_folder / f'hist_a{c_a}.png')
        plt.close()
    # Test homogeneity of variance assumption
    ratio = df.groupby([crit_a, crit_b]).std().max().values[0]
    ratio /= df.groupby([crit_a, crit_b]).std().min().values[0]
    if ratio < 2:
        print(f'Ratio between groups is less than 2 ({ratio:.2f})\n'
              f'Condition is satisfied.')
    else:
        print(f'Ratio between groups is greater or equal than 2 ({ratio:.2f})'
              f'\nCondition is NOT satisfied.')
    # Perform two-way ANOVA
    model = ols(model_str, data=df).fit()
    print(sm.stats.anova_lm(model, typ=2))
    # Generate box-plots
    with plt.style.context('seaborn-whitegrid'):
        # sns.set_style("whitegrid")
        sns.set_context("talk")
        ax = sns.boxplot(x=crit_a, y='result', hue=crit_b, data=df, notch=True)
        ax.set_xlabel(name_a)
        ax.set_ylabel('Accuracy')
        # ax.xaxis.label.set_size(15)
        # ax.yaxis.label.set_size(15)
        # ax.tick_params(labelsize=15)
        # ax.legend(title=name_b, fontsize='medium', bbox_to_anchor=(1.05, 1),
        #           loc=2, borderaxespad=0.)
        # ax.set_title(boxplot_title, fontsize=17)
        ax.legend([], [], frameon=False)
        ax.set_title(boxplot_title)
        plt.tight_layout()
        plt.savefig(out_folder / f'box_plot.png')
        plt.close()
    return df


def generate_anova_plots():
    df = anova_test('rf_results', 'rf_results_std', 'rf_anova',
                    boxplot_title='Box plot for Random Forest results')
    df = anova_test('ada_results', 'ada_results_std', 'ada_anova',
                    boxplot_title='Box plot for AdaBoost results')
    df = anova_test('bag_results', 'bag_results_std', 'bag_anova',
                    boxplot_title='Box plot for Bagging results')
    df = anova_test('svm_results', 'svm_results_std', 'svm_anova',
                    boxplot_title='Box plot for SVM results')
    df = anova_test(
        'rf_results_cmim_2', 'rf_results_std_cmim_2', 'rf_anova/cmim_2',
        boxplot_title='Box plot for Random Forest results, CMIM 2/8')
    df = anova_test(
        'rf_results_cmim_4', 'rf_results_std_cmim_4', 'rf_anova/cmim_4',
        boxplot_title='Box plot for Random Forest results, CMIM 4/8')
    df = anova_test('ada_results_cmim_2', 'ada_results_std_cmim_2',
                    'ada_anova/cmim_2',
                    boxplot_title='Box plot for AdaBoost results, CMIM 2/8')
    df = anova_test('ada_results_cmim_4', 'ada_results_std_cmim_4',
                    'ada_anova/cmim_4',
                    boxplot_title='Box plot for AdaBoost results, CMIM 4/8')
    df = anova_test('bag_results_cmim_2', 'bag_results_std_cmim_2',
                    'bag_anova/cmim_2',
                    boxplot_title='Box plot for Bagging results, CMIM 2/8')
    df = anova_test('bag_results_cmim_4', 'bag_results_std_cmim_4',
                    'bag_anova/cmim_4',
                    boxplot_title='Box plot for Bagging results, CMIM 4/8')
    df = anova_test('svm_results_cmim_2', 'svm_results_std_cmim_2',
                    'svm_anova/cmim_2',
                    boxplot_title='Box plot for SVM results, CMIM 2/8')
    df = anova_test('svm_results_cmim_4', 'svm_results_std_cmim_4',
                    'svm_anova/cmim_4',
                    boxplot_title='Box plot for SVM results, CMIM 4/8')


def plot_mask_prevalence(order_array: np.ndarray, masks: np.ndarray,
                         avg_width: int, n_parts_total: int, out_filename: str,
                         out_folder: str, title: str,
                         importances: np.ndarray = None,
                         ada_mode: bool = False):
    """Generates a plot for showing the percentage of samples that have
    masks on each feature (mask prevalence of the feature). The features
    are ordered based on the order_array, which can be from CMIM or from
    a feature importance vector.

    If an importances array is included, and depending on whether
    ada_mode is True or False, additional sorting is performed, and the
    feature importance is displayed alongside the mask prevalence.

    Parameters
    ----------
    order_array : np.ndarray
        Contains the order in which every individual feature is
        displayed. It can be a CMIM array, or an order based on
        np.argsort using feature importances.
    masks : np.ndarray
        All the masks of the dataset, in [n_samples, n_features] shape.
        It can be either the training set only, or both training and
        test.
    avg_width : int
        A moving average is displayed over the individual prevalences.
        This parameter sets the width of the moving average. For CMIM,
        40 is a good number.
    n_parts_total : int
        Used for dividing the plot into colored zones. With CMIM, I have
        been using 8. The colors match the colors used when displaying
        the location of CMIM features. If colors are not wanted, set
        this number to 0. Maximum 8.
    out_filename : str
        The name of the file to generate, without the extension. Usually
        the dataset name.
    out_folder : str
        The folder (relative or absolute) where the plots will be saved.
    title : str
        The title for the plot.
    importances : np.ndarray
        Array containing the feature importances. Set to None if unused.
    ada_mode : bool
        If True, features are sorted by importance first, prevalence
        second. Otherwise, features are sorted only by importance (and
        technically, by their original order second). Ignored if no
        importances array was given. See more details below.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # Generate out_folder
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    # Define colors
    colors = cm.get_cmap('Paired').colors
    colors = [[v for v in c] for c in colors]
    colors = colors[5::-1] + colors[-2:]
    # Calculate sum of masks
    # sum_masks = masks.sum(axis=0)[order_array]
    sum_masks = masks.sum(axis=0)
    # Generate number of features
    sum_feats = np.repeat(masks.shape[0], len(order_array))
    # Calculate prevalence for each feature
    preval = np.divide(sum_masks, sum_feats)
    xlim = len(preval) + 1
    if importances is not None and ada_mode:
        # Ada_mode requires sorting both by importance and then by mask
        # prevalence, for a more well-ordered plot, as many values are
        # repeated.
        idxs = np.argsort(preval)
        preval = preval[idxs]
        importances = importances[idxs]
        dtype = np.dtype([('importances', importances.dtype),
                          ('preval', preval.dtype)])
        struct = np.empty(len(importances), dtype=dtype)
        # Minus is added in order to sort from greater to lesser
        struct['importances'] = -importances
        struct['preval'] = preval
        idxs = np.argsort(struct, order='importances')
        importances = importances[idxs]
        preval = preval[idxs]
        # xlim = np.where(importances == 0)[0][0] + 1
        xlim = 300
    elif importances is not None:
        # Non-ada mode (i.e. RF mode) only sorts by importance, not by
        # mask size. This is the same as not including importance, but
        # this version also sorts the importances for plotting.
        idxs = np.argsort(importances)[::-1]
        importances = importances[idxs]
        preval = preval[idxs]
    else:
        # If importances are not included, the order_array is used for
        # sorting prevalence as usual.
        preval = preval[order_array]
    x = np.arange(len(preval)) + 1
    # Calculate moving average
    moving = np.convolve(preval, np.ones(avg_width), 'same') / avg_width
    # Generate plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    l1 = ax.plot(x, 100 * preval, '.',
                 label='Mask prevalence', color='tab:blue')
    l2 = ax.plot(x[:-int(avg_width / 2)], 100 * moving[:-int(avg_width / 2)],
                 label=f'Moving avg. (w={avg_width})',
                 color='tab:orange')
    areas = np.linspace(1, x.max(), n_parts_total + 1)
    for i in range(n_parts_total):
        ax.axvspan(areas[i], areas[i + 1], facecolor=colors[i], alpha=0.5)
    ax.grid('on')
    ax.set_xlabel('Feature order')
    ax.set_ylabel('Mask prevalence [%]')
    ax.set_xlim([0, xlim])
    # ax.set_xlim([0, 50])
    ax.set_ylim([0, 100 * preval[:xlim - 1].max()])
    # ax.set_ylim([0, 100 * preval[:50].max()+1])
    ax.set_title(title)
    ax.legend()
    if importances is not None:
        # Color labels and ticks blue
        ax.set_ylabel('Mask prevalence [%]', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        # Create new axis sharing x
        ax2 = ax.twinx()
        # Set new axis labels and ticks red
        ax2.set_ylabel('Feature importance', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        # Plot
        l3 = ax2.plot(x, importances,
                      label='Feature importance', color='tab:red')
        ax2.set_ylim(bottom=0)
        # Generate legend
        lns = l1 + l2 + l3
        labs = [line.get_label() for line in lns]
        ax.legend(lns, labs, loc=1)
    fig.tight_layout()
    # fig.savefig(str(out_folder / f'00000{out_filename}.png'))
    fig.savefig(str(out_folder / f'{out_filename}.png'))
    fig.clf()
    plt.close(fig)
    del sum_masks, sum_feats, preval


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
    from cmim import load_cmim_array_from_path
    from load_partitions import load_partitions_pairs
    # Define colors
    colors = cm.get_cmap('Paired').colors
    colors = [[v for v in c] for c in colors]
    colors = colors[5::-1] + colors[-2:]
    # Define folders
    cmim_folder = Path(cmim_folder)
    cmim_files = cmim_folder.glob('*.mat')
    for file in cmim_files:
        dataset_name = file.stem
        cmim_array = load_cmim_array_from_path(file)
        _, _, train_m, _, _, _, _, _ = load_partitions_pairs(
            dataset_name=dataset_name,
            partition=partition,
            mask_value=0,
            scale_dataset=True,
            pair_method=pairs
        )
        masks = train_m  # TODO quizás es mejor volver a train+test
        title = f'Mask prevalence per feature, dataset={dataset_name}'
        plot_mask_prevalence(cmim_array, masks, avg_width, n_parts_total,
                             dataset_name, out_folder, title)
