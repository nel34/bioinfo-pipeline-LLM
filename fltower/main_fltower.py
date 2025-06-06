#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main file of FLtower software.
"""

import io
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime

import fcsparser
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gmean
from tqdm import tqdm

from fltower.__version__ import __version__
from fltower.data_manager import load_parameters, save_parameters
from fltower.run_args import parse_run_args

# Suppress specific FutureWarnings from seaborn related to pandas deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Suppress Intel MKL warnings
os.environ["MKL_DISABLE_FAST_MM"] = "1"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Intel MKL.*")


class DevNull(io.IOBase):
    def write(self, *args, **kwargs):
        pass


sys.stderr = DevNull()


def read_fcs(file_path):
    try:
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
        return data, list(data.columns)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        print(f"Error type: {type(e).__name__}")
        return None, []


# Mapping labels for channels
LABEL_MAP = {
    "BL1-H": "GFP",
    "YL2-H": "RFP",
    "SSC-A": "SSC-A",
    "SSC-H": "SSC-H",
    # Add more mappings here as needed
}


def get_label(param):
    return LABEL_MAP.get(param, param)


def extract_well_key(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"([A-H])([0-9]{1,2})(?!.*[A-H][0-9]{1,2})", base_name)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return match.group(0), (letter_part, number_part)
    else:
        return base_name, (base_name, 0)


def clean_data(data, columns, remove_zeros=False):
    """Remove rows with NaN or infinite values in specified columns."""
    initial_rows = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    if remove_zeros:
        for col in columns:
            data = data[data[col] > 0]
    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with NaN or infinite values.")
    return data


def create_output_structure(results_directory):
    """
    Create necessary subdirectories within the results folder for storing different outputs.
    """
    plots_dir = os.path.join(results_directory, "plots")
    well_plots_dir = os.path.join(results_directory, "96well_plots")
    stats_dir = os.path.join(results_directory, "statistics")
    triplicate_stats_dir = os.path.join(results_directory, "triplicate_statistics")
    triplicate_plots_dir = os.path.join(results_directory, "triplicate_plots")

    for directory in [
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    return (
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    )
    
def remove_doublets(data, ssc_a="SSC-A", ssc_h="SSC-H"):
    """
    Remove doublets based on SSC-A vs SSC-H plot using vectorized operations.
    Returns the filtered data, the percentage of singlets, total events, and number of singlets.
    """
    # Filter out non-positive values
    mask = (data[ssc_a] > 0) & (data[ssc_h] > 0)
    data_filtered = data[mask]

    if len(data_filtered) == 0:
        print("Warning: No positive values found for doublet removal")
        return data, 0, len(data), 0

    ssc_ratio = data_filtered[ssc_h] / data_filtered[ssc_a]
    singlet_mask = (ssc_ratio >= 0.7) & (ssc_ratio <= 2.0)  # Adjust as needed
    singlets = data_filtered[singlet_mask]
    total_events = len(data)
    singlet_events = len(singlets)
    singlet_percentage = (singlet_events / total_events) * 100

    return singlets, singlet_percentage, total_events, singlet_events


def plot_singlet_gate(data, ssc_a="SSC-A", ssc_h="SSC-H", ax=None, file_name=None):
    """
    Plot SSC-A vs SSC-H hexbin plot with the singlet gate for original data on a given axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Filter out non-positive values
    mask = (data[ssc_a] > 0) & (data[ssc_h] > 0)
    data_filtered = data[mask]

    if len(data_filtered) == 0:
        print(f"Warning: No valid data found for {file_name}")
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return 0

    # Calculate the ratio of SSC-H to SSC-A
    ssc_ratio = data_filtered[ssc_h] / data_filtered[ssc_a]

    # Define the singlet gate
    lower_bound = 0.8
    upper_bound = 1.2

    # Create a boolean mask for singlets
    singlet_mask = (ssc_ratio >= lower_bound) & (ssc_ratio <= upper_bound)

    # Plot hexbin
    hb = ax.hexbin(
        data_filtered[ssc_a],
        data_filtered[ssc_h],
        gridsize=50,
        cmap="viridis",
        bins="log",
        xscale="log",
        yscale="log",
    )

    ax.set_xlabel(get_label(ssc_a))
    ax.set_ylabel(get_label(ssc_h))
    ax.set_title(file_name if file_name else "Singlet Gate", fontsize=8)

    # Plot the singlet gate
    x = np.logspace(
        np.log10(data_filtered[ssc_a].min()), np.log10(data_filtered[ssc_a].max()), 100
    )
    ax.plot(x, lower_bound * x, "r--", linewidth=0.5)
    ax.plot(x, upper_bound * x, "r--", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Ensure square aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(hb, cax=cax)

    # Adjust colorbar height to match the plot area
    plt.draw()  # This is necessary to update the plot layout
    ax_bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    cax.set_position(
        [cax.get_position().x0, ax_bbox.y0, cax.get_position().width, ax_bbox.height]
    )

    # Calculate and return the percentage of singlets
    singlet_percentage = (singlet_mask.sum() / len(data)) * 100
    ax.text(
        0.05,
        0.95,
        f"Singlets: {singlet_percentage:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",
    )

    return singlet_percentage