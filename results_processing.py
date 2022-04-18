import pickle
from pathlib import Path
# from textwrap import fill
from typing import Tuple, List, Union

import numpy as np
import pandas as pd

from utils import find_shape


def load_accuracies_from_results(results_folder: Union[str, Path]):
    """Loads accuracies from .pickle images, returns them in a dictionary
    where the keys are the stems of the images.
    """
    results_folder = Path(results_folder)
    if not results_folder.exists():
        raise ValueError(f'{results_folder} not found')
    accuracies = {}
    for file in results_folder.glob('*.pickle'):
        with open(file, 'rb') as f:
            cur_results = pickle.load(f)
        cur_results = np.array([d['accuracy'] for d in cur_results])
        accuracies[file.stem] = cur_results
    return accuracies


def format_results_as_str(results_arr: np.array):
    """Formats results as mean ± std"""
    mean_value = results_arr.mean()
    std_value = results_arr.std()
    if results_arr.max() <= 1:
        mean_value *= 100
        std_value *= 100
    formatted = f'{mean_value:.2f} ± {std_value:.2f}'
    return formatted


def generate_boxplots(results_folder: str, out_file, title: str):
    """For reviewing VGG results"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_folder = Path(results_folder)
    results = load_accuracies_from_results(results_folder)

    df = pd.DataFrame(results)
    with sns.axes_style('whitegrid'), sns.plotting_context('notebook',
                                                           font_scale=1.2):
        ax = sns.boxplot(data=df, notch=True)
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        plt.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file)
    return results


def generate_table_from_df(
        input_df: pd.DataFrame, annot_df: pd.DataFrame, out_folder: str,
        out_name: str, figsize: Tuple[float] = (18, 6), display: bool = False
):
    """Generates and saves a table, generated from two dataframes. The
    first one contains the numeric results (for coloring the cells) and
    the second one contains the annotations (for cell texts).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=1.5)
    # Generate image
    ax = sns.heatmap(input_df, annot=annot_df, fmt='', cbar=False,
                     linewidths=1, linecolor='black')
    ax.xaxis.tick_top()
    plt.tight_layout()
    # Save image
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_folder / out_name)
    if display:
        plt.show()
    plt.clf()


def generate_results_table_from_folders(
        folders: List[str], folder_prefixes: List[str], out_folder: str,
        out_name: str, figsize: Tuple[float] = (18, 6),
        display: bool = False, transpose: bool = False
):
    """Using the selected results folders, generates a formatted table
    as an image with all the results, ready for adding to the ppt.
    """
    # Prepare folders
    folders = [Path(f) for f in folders]
    if any(not f.exists() for f in folders):
        bad = [f for f in folders if not f.exists()]
        bad = ", ".join([f.name for f in bad])
        raise Exception(f'Folder not found: {bad}')
    # Process folders results
    results_str = {}
    results = {}
    for i, folder in enumerate(folders):
        folder_results_str = {}
        folder_results = {}
        cur_pref = folder_prefixes[i]
        accuracies = load_accuracies_from_results(
            results_folder=folder
        )
        for cur_key, acc in accuracies.items():
            folder_results_str[cur_pref + cur_key] = format_results_as_str(acc)
            folder_results[cur_pref + cur_key] = acc.mean()
        results.update(folder_results)
        results_str.update(folder_results_str)
    # Generate DF with results
    df = pd.DataFrame(results, index=[0])
    annot_df = pd.DataFrame(results_str, index=[0])
    if transpose:
        df = df.transpose()
        annot_df = annot_df.transpose()
    # Generate image
    generate_table_from_df(
        input_df=df,
        annot_df=annot_df,
        out_folder=out_folder,
        out_name=out_name,
        figsize=figsize,
        display=display
    )

    return df, results, results_str


def merge_vgg_full_results(results_folder):
    """VGG results are currently dumped on separate files, due to the
    tests being run using a bash file. This function merges the results
    of the same test but different partitions together.
    """
    results_folder = Path(results_folder)
    unpacked_folder = results_folder / 'unpacked'
    unpacked_folder.mkdir(exist_ok=True)
    # Find prefixes
    prefixes = []
    for rf in results_folder.glob('*.pickle'):
        prefix = '_'.join(rf.stem.split('_')[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)
    # Gather results for each prefix
    for prefix in prefixes:
        results = []
        globs = [*list(results_folder.glob(f'{prefix}_?.pickle')),
                 *list(results_folder.glob(f'{prefix}_??.pickle'))]
        for rf in globs:
            with open(rf, 'rb') as f:
                cur_res = pickle.load(f)
            if isinstance(cur_res, list):
                cur_res = cur_res[0]
            results.append(cur_res)
            rf.rename(unpacked_folder / rf.name)
        with open(results_folder / f'{prefix}.pickle', 'wb') as f:
            pickle.dump(results, f)


def generate_cv_grid_plot(dataset_name: str, results_folder: str,
                          partition: int, text_color='black'):
    """Generates visualizations of the cross-validation results for the
    chosen dataset.
    """
    from utils import grid_plot
    cv_file = Path(results_folder) / f'{dataset_name}.pickle'
    if not cv_file.exists():
        raise FileNotFoundError(f'{results_folder}/{cv_file.name} not found')
    with open(cv_file, 'rb') as f:
        cv_results: dict = pickle.load(f)[partition]['cv_results']
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
            i = b + n_b * a
            results_grid[a, b] = results[i]
            std_grid[a, b] = results_std[i]
    # Generate results plot
    fig, ax = grid_plot(list_a, list_b, results_grid * 100,
                        text_color=text_color)
    ax.set_ylabel(name_a)
    ax.set_xlabel(name_b)
    ax.set_title(f'{dataset_name}_{partition}, CV results')
    # Generate results std plot
    fig_std, ax_std = grid_plot(list_a, list_b, std_grid * 100)
    ax_std.set_ylabel(name_a)
    ax_std.set_xlabel(name_b)
    ax_std.set_title(f'{dataset_name}_{partition}, CV results, std')
    return fig, fig_std


def review_vgg_step_by_step(dataset_name: str, results_folder, title,
                            out_file, avg_eval=False, peri_mode=False):
    """Generates a training curve plot from the step-by-step results"""
    from itertools import product
    import matplotlib.pyplot as plt
    font = {'weight': 'bold',
            'size': 16}
    plt.rc('font', **font)
    # Load results
    results_folder = Path(results_folder)
    if not peri_mode:
        results_file = list(results_folder.glob(f'{dataset_name}_?.pickle'))[0]
    else:
        results_file = list(results_folder.glob(
            f'{dataset_name}_?/callback_results.pickle'))[0]
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    # Extract results
    names = list(results[0].keys())
    n_names = len(names)
    accuracies = np.zeros((n_names, len(results)))
    for i, j in product(range(n_names), range(len(results))):
        cur_name = names[i]
        cur_results = results[j][cur_name]
        accuracies[i, j] = cur_results['accuracy'] * 100
    # Plot results
    fig, ax = plt.subplots()
    x = np.arange(len(results))
    if not avg_eval:
        for i in range(n_names):
            ax.plot(x, accuracies[i, :], label=names[i], linewidth=2)
    else:
        ax.plot(x, accuracies[0, :], label=names[0], linewidth=2)
        ax.plot(x, accuracies[1:3, :].mean(axis=0), label='eval', linewidth=2)

    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(title)
    ax.set_ylim([48, 100])
    plt.grid('on')
    plt.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_file)
    plt.show()
    plt.clf()
    plt.close()


def generate_pair_visualization(iris_a, mask_a, iris_b, mask_b):
    orig_shape = find_shape(len(iris_a))

    def retrieve_reshape_rgb(iris, to_rgb=False):
        """Retrieves the iris or mask, and reshapes to its original
        shape. If to_rgb, the array is turned to RGB (3 channels).
        """
        out_array = iris.reshape(orig_shape)
        if to_rgb:
            return np.tile(out_array[..., np.newaxis], (1, 1, 3))
        return out_array

    # Reshape current iris into RGB rectangular images
    iris_a = retrieve_reshape_rgb(iris_a, to_rgb=True)
    iris_b = retrieve_reshape_rgb(iris_b, to_rgb=True)

    # Generate visualization of original masks
    mask_a = retrieve_reshape_rgb(mask_a)
    mask_b = retrieve_reshape_rgb(mask_b)
    iris_a_pre = iris_a.copy()
    iris_a_pre[mask_a == 1] = [255, 0, 255]  # Magenta
    iris_b_pre = iris_b.copy()
    iris_b_pre[mask_b == 1] = [255, 0, 255]  # Magenta

    # Generate visualization of paired masks
    # - mask_ab_x is the union of the masks minus mask x
    mask_ab_a = mask_b.copy()
    mask_ab_a[mask_a == 1] = 0
    iris_a_post = iris_a_pre.copy()
    iris_a_post[mask_ab_a == 1] = [0, 255, 0]  # Green
    mask_ab_b = mask_a.copy()
    mask_ab_b[mask_b == 1] = 0
    iris_b_post = iris_b_pre.copy()
    iris_b_post[mask_ab_b == 1] = [0, 255, 0]  # Green

    images = [iris_a_pre, iris_a_post, iris_b_pre, iris_b_post]

    return images


def save_pairs_visualizations(pairs, data_x, data_m, out_folder: str,
                              to_visualize: list = None, pair_scores=None):
    """Visualize mask pairs by generating an RGB image displaying the
    masks before and after pairing, using different colors for each one.

    Parameters
    ----------
    pairs : np.array
        Array containing the pairs, as returned by SPPMat.generate_pairs
    data_x, data_m : np.array
        Arrays containing the iris data and original masks respectively
    to_visualize : list of ints, optional
        List containing the indexes of the pairs to be visualized.
        If None, 5 pairs are selected at random.
    pair_scores : np.array
        SPP score of each pair
    """
    from PIL import Image
    if to_visualize is None:
        to_visualize = np.random.randint(pairs.shape[1], size=5)
    # Initialize out folder
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Scale data and turn to uint8
    if data_x.max() == 1:
        data_x *= 255
    data_x = data_x.astype('uint8')

    for pair_idx in to_visualize:
        [idx_a, idx_b] = pairs[:, pair_idx]
        iris_a = data_x[idx_a, :]
        mask_a = data_m[idx_a, :]
        iris_b = data_x[idx_b, :]
        mask_b = data_m[idx_b, :]

        images = generate_pair_visualization(iris_a, mask_a, iris_b, mask_b)
        # names = ['iris_a_pre', 'iris_a_post', 'iris_b_pre', 'iris_b_post']

        images = np.vstack(images)
        img = Image.fromarray(images)
        if pair_scores is None:
            out_name = f'{pair_idx}.png'
        else:
            cur_score = pair_scores[pair_idx] * 1000
            out_name = f'{cur_score:0.0f}_{pair_idx}.png'
        img.save(out_folder / out_name)


def analize_pairs(pair_scores, bad_score=0.1, delta=0.01):
    def calculate_histogram():
        """This version treats bin edges opposite of numpy: first bin
        includes both edges, and from the second bin on it excludes the
        left edge and includes the right edge. This is done to ensure
        that pairs strictly over bad_score are not grouped with pairs
        that are exactly the bad_score (which is not bad)."""
        _bins = np.arange(0, bad_score + delta, delta)
        _bins = np.hstack([_bins, [1]])
        n_bins = len(_bins)
        _hist = []
        for i in range(n_bins - 1):
            """
            [0,  0.01]
            ]0.01 0.02]
            ...
            ]0.09 0.1]
            ]0.1  1.0]  # Bad pairs
            """
            if i == 0:
                lower = _bins[i] <= pair_scores
            else:
                lower = _bins[i] < pair_scores
            upper = pair_scores <= _bins[i+1]
            _hist.append(np.sum(lower & upper))
        return _hist, _bins

    # Obtain histogram
    histogram, bins = calculate_histogram()

    # Sum of good pairs scores
    good_scores_idx = pair_scores <= bad_score
    good_scores = pair_scores[good_scores_idx]
    sum_good_scores = np.sum(good_scores)
    n_good_scores = np.sum(good_scores_idx)

    avg_good_score = sum_good_scores / n_good_scores

    # Count of bad pairs
    n_bad_pairs = histogram[-1]

    to_return = {
        'histogram': histogram,
        'bins': bins,
        'avg_good_score': avg_good_score,
        'n_bad_pairs': n_bad_pairs
    }

    return to_return


def plot_pairs_histogram(hist, bins, out_folder, dataset, threshold,
                         max_y=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    with sns.plotting_context('talk'):
        colors = sns.color_palette()[:2]

        norm_hist = 100 * hist / np.sum(hist)
        x = np.array(bins[:-1])
        x = 100*(x + (x[0] + x[1]) / 2)
        delta = (x[0] + x[1]) / 2
        ticks = x - delta/2
        labels = [f'{v:.1f}' for v in ticks]
        bar_colors = [colors[0]] * (len(norm_hist) - 1)
        bar_colors.append(colors[1])
        ax = plt.bar(x, norm_hist, width=delta * 0.9, color=bar_colors,
                     edgecolor='black', linewidth=2)
        plt.bar_label(ax, hist)
        plt.xticks(ticks[::2], labels[::2])
        plt.grid(True, axis='y')
        plt.ylabel('% of pairs')
        plt.xlabel('% of growth')
        if max_y is not None:
            plt.ylim([0, max_y])
        plt.title(f'Pairs distrib., {dataset}, thresh.={threshold * 100:.1f}')
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset}_{threshold*100:.1f}.png')
        plt.clf()


def plot_pairs_analysis(thresholds, avg_good_scores, n_bad_pairs, out_folder,
                        dataset_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        'threshold': thresholds,
        'avg_good_score': np.array(avg_good_scores) * 100,
        'n_bad_pairs': n_bad_pairs
    })
    with sns.plotting_context('talk'):
        colors = sns.color_palette()[:2]
        ax1 = df.plot(
            x='threshold', y='avg_good_score', color=colors[0], legend=False
        )
        ax1.set_ylabel('Avg. growth [%]')
        ax1.yaxis.label.set_color(colors[0])
        ax2 = plt.twinx()
        df.plot(
            x='threshold', y='n_bad_pairs', color=colors[1], legend=False,
            ax=ax2
        )
        ax2.set_ylabel('N. of bad pairs')
        ax2.yaxis.label.set_color(colors[1])
        plt.title(f'Pairs analysis, {dataset_name}')
        ax1.figure.legend(loc='lower right', markerscale=0.5,
                          fontsize='x-small')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_folder / f'{dataset_name}.png')
        plt.clf()


def process_pairs_thresh_results_to_df(results_folder):
    """Process the results inside the folder into a DataFrame, with
    columns for the dataset, threshold and accuracy.

    The folder is expected to contain more folders, each with a
    threshold set as name. Within each folder should be the .pickle
    files with the results for each dataset.
    """
    results = {}
    results_folder = Path(results_folder)
    for threshold_folder in results_folder.glob('*/'):
        threshold = threshold_folder.name
        for results_file in threshold_folder.glob('*.pickle'):
            dataset = results_file.stem
            cur_results = np.load(results_file, allow_pickle=True)
            cur_results = [r['accuracy'] for r in cur_results]
            if dataset not in results:
                results[dataset] = {threshold: cur_results}
            else:
                results[dataset][threshold] = cur_results
    df = pd.DataFrame(results)
    df['threshold'] = df.index.map(lambda x: float(x))
    df = df.melt(id_vars=['threshold'], value_vars=df.columns[:-1],
                 var_name='dataset', value_name='accuracy')
    df = df.explode('accuracy', ignore_index=True)
    df.dataset = df.dataset.astype('category')
    df.accuracy = df.accuracy.astype('float64')
    return df


def plot_pairs_thresh_results(
        results_folder, out_file,
        title='VGG+LSVM results using pairs, variable threshold',
        ylim=(0.53, 0.64)):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = process_pairs_thresh_results_to_df(results_folder)

    with sns.axes_style('whitegrid', rc={'xtick.bottom': True,
                                         'ytick.left': True}), \
            sns.plotting_context('notebook', font_scale=1.4):
        ax = sns.lmplot(data=df, x='threshold', y='accuracy', hue='dataset',
                        col='dataset', x_estimator=np.mean, col_wrap=2,
                        height=4, aspect=1.4)
        ax.fig.subplots_adjust(top=0.9)
        ax.fig.suptitle(title)
        plt.ylim(ylim)
        plt.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file)
        plt.clf()

    return df


def anova_test(
        folders_list: list, diff_var_name: str, diff_var_list: list,
        out_folder, crit_a='fixed', crit_b=None, name_a='Fixed masks',
        name_b='Pair deletion thresh.', boxplot_title=None):
    """Performs a two-way ANOVA test on the results of each folder.
    Folders should contain the same type of results. The name of the
    variable that differentiates should be set, and the value of this
    variable should be indicated in a list.
    """
    from itertools import product
    from bioinfokit.analys import stat as bioinfokit_stat
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    import scipy.stats
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    crit_b = diff_var_name if crit_b is None else crit_b

    dfs_list = []
    for i, f in enumerate(folders_list):
        cur_df = process_pairs_thresh_results_to_df(f)
        cur_df[diff_var_name] = diff_var_list[i]
        dfs_list.append(cur_df)

    df = pd.concat(dfs_list, ignore_index=True)
    df['fixed'] = df.dataset.apply(lambda x: x.endswith('fixed'))

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Test whether samples follow normal distribution with Q-Q plot
    crit_a_values = df[crit_a].unique()
    crit_b_values = df[crit_b].unique()
    for c_a, c_b in product(crit_a_values, crit_b_values):
        condition = (df[crit_a] == c_a) & (df[crit_b] == c_b)
        fig, ax = plt.subplots()
        scipy.stats.probplot(df[condition]['accuracy'], dist='norm', plot=ax)
        ax.set_title(f'Probability plot - {name_a}: {c_a}, {name_b}: {c_b}',
                     fontsize=15)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        fig.tight_layout()
        fig.savefig(out_folder / f'qq_a{str(c_a)}_b{str(c_b)}.png')
        plt.close()

    # Test whether samples follow normal distribution with Shapiro-Wilk
    res = bioinfokit_stat()
    model_str = f'accuracy ~ C({crit_a}) + C({crit_b}) + ' \
                f'C({crit_a}):C({crit_b})'
    res.tukey_hsd(
        df=df, res_var='accuracy', xfac_var=[crit_a, crit_b],
        anova_model=model_str
    )
    _, pvalue = scipy.stats.shapiro(res.anova_model_out.resid)
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
        with sns.plotting_context('talk'):
            ax = sns.histplot(data=cur_df, x='accuracy', hue=crit_b,
                              binwidth=0.01, binrange=(min_val, max_val))
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
        with sns.plotting_context('talk'):
            ax = sns.boxplot(x=crit_a, y='accuracy',
                             hue=crit_b, data=df, notch=True)
            ax.set_xlabel(name_a)
            ax.set_ylabel('Accuracy')
            # ax.xaxis.label.set_size(15)
            # ax.yaxis.label.set_size(15)
            # ax.tick_params(labelsize=15)
            # ax.legend(title=name_b, fontsize='medium',
            #           bbox_to_anchor=(1.05, 1),
            #           loc=2, borderaxespad=0.)
            # ax.set_title(boxplot_title, fontsize=17)
            ax.legend([], [], frameon=False)
            ax.set_title(boxplot_title)
            plt.tight_layout()
            plt.savefig(out_folder / 'box_plot.png')
            plt.close()

    return df
