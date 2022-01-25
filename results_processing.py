import pickle
from pathlib import Path
from textwrap import fill
from typing import Tuple, List, Union

import numpy as np
import pandas as pd


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
