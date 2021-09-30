from pathlib import Path
import pickle

import numpy as np

from constants import datasets
from utils import grid_plot, find_dataset_shape


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


def review_results(results_folder: str):
    results_folder = Path(results_folder)
    for file in results_folder.glob('*.pickle'):
        with open(file, 'rb') as f:
            cur_results = pickle.load(f)
        cur_results = np.array([d['accuracy'] for d in cur_results])
        mean = cur_results.mean() * 100
        std = cur_results.std() * 100
        print(f'{file.stem}:\t{mean:.2f} ± {std:.2f}')


def generate_mask_visualization(dataset_name: str, pairs: str,
                                partition=1):
    """Generates a grayscale visualization of the masks of the dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use
    pairs : str or None/False
        Set to None/False if pairs are not to be used. Otherwise, set to the
        pairing method name.
    partition : int
        Train partition to use. Default 1.
    """
    from load_partitions import load_partitions_pairs
    _, _, train_m, _, _, _, _, _ = load_partitions_pairs(
        dataset_name, partition, 0, True, pairs)
    masks = train_m.mean(axis=0)
    masks = masks * 255
    shape = find_dataset_shape(dataset_name)
    masks = masks.reshape(shape)
    return masks.astype('uint8')


def visualize_all_masks(out_folder: str, use_pairs: bool):
    """Generates visualizations for all masks"""
    import matplotlib.pyplot as plt
    from constants import PAIR_METHOD
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
               crit_a='fixed', crit_b='uses_pairs'):
    """Performs a two-way ANOVA test on the results obtained for this
     classifier. The variables to compare are pairing and mask-fixing.
     """
    from itertools import product
    from bioinfokit.analys import stat
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
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
        ax.set_title(f"Probability plot - {crit_a}: {c_a}, {crit_b}: {c_b}",
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
    return df
