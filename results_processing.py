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
    
    # Obtain original shape of data
    orig_shape = find_shape(n_features=data_x.shape[1])
    # Scale data and turn to uint8
    if data_x.max() == 1:
        data_x *= 255
    data_x = data_x.astype('uint8')

    def retrieve_reshape_rgb(array, index, to_rgb=False):
        """Retrieves the iris or mask, and reshapes to its original
        shape. If to_rgb, the array is turned to RGB (3 channels).
        """
        out_array = array[index, :].reshape(orig_shape)
        if to_rgb:
            return np.tile(out_array[..., np.newaxis], (1, 1, 3))
        return out_array

    for pair_idx in to_visualize:
        [idx_a, idx_b] = pairs[:, pair_idx]
        iris_a = data_x[idx_a, :]
        mask_a = data_m[idx_a, :]
        iris_b = data_x[idx_b, :]
        mask_b = data_m[idx_b, :]

        images = generate_pair_visualization(iris_a, mask_a, iris_b, mask_b)
        names = ['iris_a_pre', 'iris_a_post', 'iris_b_pre', 'iris_b_post']

        # for img, name in zip(images, names):
        #     img = Image.fromarray(img)
        #     if pair_scores is None:
        #         out_name = f'{pair_idx}_{name}.png'
        #     else:
        #         cur_score = pair_scores[pair_idx] * 100
        #         out_name = f'{cur_score:0.0f}_{pair_idx}_{name}.png'
        #     img.save(out_folder / out_name)

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

