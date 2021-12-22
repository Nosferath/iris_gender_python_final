from pathlib import Path, PurePath
import pickle
from textwrap import fill
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

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
            i = b + n_b * a
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


def load_accuracies_from_results(results_folder: Union[str, Path],
                                 include_train_acc: bool = False):
    """Loads accuracies from .pickle files, returns them in a dictionary
    where the keys are the stems of the files.
    """
    results_folder = Path(results_folder)
    if not results_folder.exists():
        raise ValueError(f'{results_folder} not found')
    accuracies = {}
    train_acc = {}
    for file in results_folder.glob('*.pickle'):
        with open(file, 'rb') as f:
            cur_results = pickle.load(f)
        if 'acc_train' in cur_results[0] and include_train_acc:
            train_results = np.array([d['acc_train'] for d in cur_results])
            train_acc[file.stem] = train_results
        cur_results = np.array([d['accuracy'] for d in cur_results])
        accuracies[file.stem] = cur_results
    if include_train_acc:
        return accuracies, train_acc
    return accuracies


def format_results_as_str(results_arr: np.ndarray):
    """Formats results as mean ± std"""
    mean_value = results_arr.mean()
    std_value = results_arr.std()
    if results_arr.max() <= 1:
        mean_value *= 100
        std_value *= 100
    formatted = f'{mean_value:.2f} ± {std_value:.2f}'
    return formatted


def review_results(results_folder: Union[str, Path], print_train=False):
    """Prints results in a readable way."""
    accuracies, train_acc = load_accuracies_from_results(results_folder, True)
    for key, cur_results in accuracies.items():
        if print_train and key in train_acc:
            train_results = train_acc[key]
            formatted = format_results_as_str(train_results)
            print(f'Train {key}:\t{formatted}')
        formatted = format_results_as_str(cur_results)
        print(f'{key}:\t{formatted}')


def generate_table_from_df(input_df: pd.DataFrame, annot_df: pd.DataFrame,
                           out_folder: str, out_name: str,
                           figsize: Tuple[float] = (18, 6),
                           display: bool = False):
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
        folders: List[str], keys: List[str],
        out_folder: str, out_name: str,
        figsize: Tuple[float] = (18, 6),
        display: bool = False, include_train: bool = False,
        transpose: bool = False
):
    """Using the selected results folders, generates a formatted table
    as an image with all the results, ready for adding to the ppt.
    """
    # Prepare folders
    if len(folders) != len(keys):
        raise Exception('Length of folders and keys must match')
    folders = [Path(f) for f in folders]
    if any(not f.exists() for f in folders):
        bad = [f for f in folders if not f.exists()]
        bad = ", ".join([f.name for f in bad])
        raise Exception(f'Folder not found: {bad}')
    # Process folders results
    results_str = {}
    results = {}
    for i, folder in enumerate(folders):
        key = fill(keys[i], 12)  # Classifier
        folder_results_str = {}
        folder_results = {}
        accuracies, train_acc = load_accuracies_from_results(
            results_folder=folder,
            include_train_acc=True
        )
        for cur_key, acc in accuracies.items():
            if include_train and len(train_acc):
                cur_train_acc = train_acc[cur_key]
                folder_results_str[cur_key + '_Train'] = \
                    format_results_as_str(cur_train_acc)
                folder_results[cur_key + '_Train'] = cur_train_acc.mean()
            folder_results_str[cur_key] = format_results_as_str(acc)
            folder_results[cur_key] = acc.mean()
        results[key] = folder_results
        results_str[key] = folder_results_str
    # Generate DF with results
    df = pd.DataFrame(results)
    annot_df = pd.DataFrame(results_str)
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

    # return df, results, results_str


def generate_roc_curve(model, test_x: np.ndarray, test_y: np.ndarray,
                       title='Curva ROC, XGBoost en Dataset Higgs-1M'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    pred = model.predict_proba(test_x)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, pred)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, 'b',
             label=f'Curva ROC, (AUC={auc_score:0.4f})')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.tight_layout()
    plt.show()


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


def generate_training_curves(model, out_file: Union[str, PurePath],
                             model_name: str = 'XGBoost'):
    """Generates plots for evaluating early stop and overfitting:
    Classification error and logloss plots, in train vs eval.

    Parameters
    ==========
    model : trained model such as XGBoost
        Model that implements the evals_result method and that has been
        evaluated with a validation partition during training.
    out_file : str or PurePath
        Path and name (prefix) of the files where the plots will be
        saved to. Non-existant paths will be created. If out_file has a
        file extension it will be ignored, as the file name is only used
        as a prefix.
    model_name : str (optional)
        Name of the model. Used as a title prefix. Default: XGBoost.
    """
    assert hasattr(model, 'evals_result'), 'Model does not implement' \
                                           ' evals_result.'
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Parse out filenames from out_file
    out_file = Path(out_file)
    root_path = out_file.parent
    root_path.mkdir(exist_ok=True, parents=True)
    logloss_path = root_path / f'{out_file.stem}_logloss.png'
    error_path = root_path / f'{out_file.stem}_error.png'
    # Get values from model
    evals = model.evals_result()
    epochs = len(evals['validation_0']['error'])
    x_axis = range(0, epochs)
    # Generate plots
    for_values = (('logloss', logloss_path, 'Log Loss'),
                  ('error', error_path, 'Classification Error'))
    for key, filename, y_label in for_values:
        with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
            df = pd.DataFrame({
                'epochs': x_axis,
                'Train': evals['validation_0'][key],
                'Valid.': evals['validation_1'][key],
            })
            df = df.melt(id_vars='epochs', value_vars=['Train', 'Valid.'],
                         var_name='set', value_name=y_label)
            fig, ax = plt.subplots()
            sns.lineplot(x='epochs', y=y_label, hue='set', data=df, ax=ax)
            ax.legend()
            plt.ylabel(y_label)
            plt.title(f'{model_name} {y_label}')
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()


def generate_feature_selection_plot(results: dict, results_early: dict,
                                    out_folder: Union[str, PurePath],
                                    sel_name: str = 'XGBoost'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Generate DataFrames and remove unnecesary data
    df = pd.DataFrame(results)
    df = df.drop(['0', '1', 'macro avg', 'weighted avg'], axis=0)
    df = df.transpose()
    df_early = pd.DataFrame(results_early)
    df_early = df_early.drop(['0', '1', 'macro avg', 'weighted avg'], axis=0)
    df_early = df_early.transpose()
    # Rename columns and merge
    df.columns = ['normal', 'n_feats']
    df.index.name = 'fscore threshold'
    df_early.columns = ['early', 'n_feats']
    df_early.index.name = 'fscore threshold'
    df = pd.merge(df, df_early, on=['fscore threshold', 'n_feats'])
    df = df.reset_index()
    # Melt df for plotting
    df = df.melt(
        id_vars=['n_feats', 'fscore threshold'],
        value_vars=['normal', 'early'],
        var_name='test type', value_name='accuracy'
    )
    sns.set_style('whitegrid')
    sns.set_context('talk')
    ax = sns.lineplot(data=df, x='n_feats', y='accuracy', hue='test type')
    plt.legend()
    ax.set_xlabel('N. Features')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(f'Feature selection, {sel_name}, dataset '
                 f'{out_folder.name.upper()}')
    plot_name = f'{out_folder.name}_plot.png'
    plt.tight_layout()
    plt.savefig(out_folder / plot_name)
    plt.clf()


def generate_df_from_xgb_param_cv_results(folder, results_file_name):
    dfs = []
    folder = Path(folder)
    for dataset_folder in [d for d in folder.glob('*/') if d.is_dir()]:
        # Load cv results
        with open(dataset_folder / results_file_name, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                cv_results = loaded['cv_results']
            else:
                cv_results = loaded.cv_results_
        # Determine params and splits
        params = [k for k in cv_results.keys()
                  if k.startswith('param_')]
        splits = [k for k in cv_results.keys()
                  if k.startswith('split') and k.endswith('_test_score')]
        # Generate df
        df = pd.DataFrame(cv_results)
        df = df.melt(id_vars=params, value_vars=splits, var_name='split',
                     value_name='accuracy')
        df.loc[:, 'split'] = df.loc[:, 'split'].apply(lambda x: int(x[5]))
        df = df.sort_values(params, axis=0)
        df['dataset'] = dataset_folder.name
        # Fix learning rate
        if 'param_learning_rate' in params:
            df.param_learning_rate = \
                df.param_learning_rate.astype(float).round(3)
        dfs.append(df)

    return dfs
