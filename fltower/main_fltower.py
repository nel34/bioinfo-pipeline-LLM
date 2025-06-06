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