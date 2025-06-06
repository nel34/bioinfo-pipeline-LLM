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

def plot_histogram(
    singlets,
    x_param,
    file_name,
    ax,
    x_scale="linear",
    kde=False,
    color="blue",
    xlim=None,
    gates=None,
):
    """
    Plot a histogram of the singlets data for a given parameter and calculate statistics.
    """
    cleaned_data = clean_data(singlets, [x_param])
    if x_scale == "log" and cleaned_data[x_param].min() <= 0:
        cleaned_data = cleaned_data[cleaned_data[x_param] > 0]

    sns.histplot(
        cleaned_data[x_param],
        bins=100,
        kde=kde,
        ax=ax,
        log_scale=(x_scale == "log"),
        color=color,
    )

    if xlim:
        ax.set_xlim(xlim)

    # Keep log ticks if x_scale is 'log'
    if x_scale == "log":
        ax.set_xscale("log")

        # Custom formatter function
        def log_tick_formatter(x, pos):
            return f"$10^{{{int(np.log10(x))}}}$"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))

    # Calculate global statistics
    global_gm = gmean(cleaned_data[x_param])
    global_median = cleaned_data[x_param].median()
    num_events = len(cleaned_data)

    stats = {
        "Global_GM": global_gm,
        "Global_Median": global_median,
        "Num_Events": num_events,
    }

    if gates:
        gate_colors = plt.cm.rainbow(np.linspace(0, 1, len(gates)))
        y_max = ax.get_ylim()[1]
        for i, ((gate_min, gate_max), gate_color) in enumerate(zip(gates, gate_colors)):
            ax.axvline(gate_min, color=gate_color, linestyle="--")
            ax.axvline(gate_max, color=gate_color, linestyle="--")

            gate_data = cleaned_data[
                (cleaned_data[x_param] >= gate_min)
                & (cleaned_data[x_param] <= gate_max)
            ]
            percentage = (len(gate_data) / num_events) * 100
            gate_gm = gmean(gate_data[x_param]) if len(gate_data) > 0 else 0

            stats[f"Gate_{i+1}_Percentage"] = percentage
            stats[f"Gate_{i+1}_GM"] = gate_gm

            # Add gate label with percentage
            gate_center = (gate_min + gate_max) / 2
            y_pos = y_max * (0.95 - i * 0.1)  # Adjust vertical position for each gate
            ax.text(
                gate_center,
                y_pos,
                f"Gate {i+1}: {percentage:.2f}%",
                color=gate_color,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor=gate_color, alpha=0.7, pad=2),
            )

    ax.set_xlabel(get_label(x_param))
    ax.set_ylabel("Count")
    ax.set_title(
        f"{file_name} - {get_label(x_param)}", fontsize=10, fontweight="bold", pad=20
    )

    return stats


def compute_statistics(data):
    """
    Calculate mean, geometric mean, and median for the data.
    """
    if not data.empty:
        mean = data.mean()
        geometric_mean = gmean(
            data[data > 0]
        )  # geometric mean only for positive values
        median = data.median()
    else:
        mean = geometric_mean = median = np.nan
    return mean, geometric_mean, median


def calculate_gate_statistics(data, gate_min, gate_max):
    """
    Calculate the percentage and geometric mean of cells within a specified gate.
    """
    total_cells = len(data)
    cells_in_gate = data[(data >= gate_min) & (data <= gate_max)]
    percentage = (len(cells_in_gate) / total_cells) * 100
    geo_mean = gmean(cells_in_gate) if len(cells_in_gate) > 0 else 0
    return percentage, geo_mean


def plot_scatter_with_manual_gates(
    data,
    x_param,
    y_param,
    file_name,
    ax,
    scatter_type="scatter",
    cmap="viridis",
    x_scale="linear",
    y_scale="linear",
    xlim=None,
    ylim=None,
    gridsize=100,
    quadrant_gates=None,
):
    print(f"Plotting {scatter_type} scatter with manual gates for {file_name}")
    print(
        f"Quadrant gates: {quadrant_gates}"
    )  # Add this line to print the quadrant gates

    # Check if both parameters exist in the data
    if x_param not in data.columns or y_param not in data.columns:
        print(
            f"Error: One or both parameters ({x_param}, {y_param}) not found in the data for {file_name}"
        )
        return None

    # Clean the data
    cleaned_data = clean_data(data, [x_param, y_param])

    # Downsample if there are too many points
    if len(cleaned_data) > 1000000:
        cleaned_data = cleaned_data.sample(n=1000000, random_state=42)

    # Handle non-positive values for log scale
    if x_scale == "log":
        cleaned_data[x_param] = cleaned_data[x_param].clip(lower=1)
    if y_scale == "log":
        cleaned_data[y_param] = cleaned_data[y_param].clip(lower=1)

    if scatter_type == "density":
        # Plot hexbin
        hb = ax.hexbin(
            cleaned_data[x_param],
            cleaned_data[y_param],
            gridsize=gridsize,
            cmap=cmap,
            xscale=x_scale,
            yscale=y_scale,
            bins="log",
            mincnt=1,
            rasterized=True,
        )

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(hb, cax=cax, label="Count")
    else:
        # Plot scatter
        ax.scatter(
            cleaned_data[x_param], cleaned_data[y_param], c="blue", s=0.1, alpha=0.5
        )

    ax.set_xlabel(get_label(x_param))
    ax.set_ylabel(get_label(y_param))
    ax.set_title(os.path.basename(file_name), fontsize=12, fontweight="bold")

    # Set scales and keep log ticks
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    # Custom formatter function
    def log_tick_formatter(x, pos):
        return f"$10^{{{int(np.log10(x))}}}$"

    if x_scale == "log":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))
    if y_scale == "log":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Ensure square aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Use quadrant gates if provided, otherwise use median values
    if quadrant_gates and "x" in quadrant_gates and "y" in quadrant_gates:
        x_mid = quadrant_gates["x"]
        y_mid = quadrant_gates["y"]
        print(f"Using provided quadrant gates: x={x_mid}, y={y_mid}")  # Add this line
    else:
        x_mid = np.median(cleaned_data[x_param])
        y_mid = np.median(cleaned_data[y_param])
        print(
            f"Using median values for quadrant gates: x={x_mid}, y={y_mid}"
        )  # Add this line

    # Define quadrants
    quadrants = {
        "Q1": (cleaned_data[x_param] >= x_mid) & (cleaned_data[y_param] >= y_mid),
        "Q2": (cleaned_data[x_param] < x_mid) & (cleaned_data[y_param] >= y_mid),
        "Q3": (cleaned_data[x_param] < x_mid) & (cleaned_data[y_param] < y_mid),
        "Q4": (cleaned_data[x_param] >= x_mid) & (cleaned_data[y_param] < y_mid),
    }

    # Calculate percentages for each quadrant
    total_cells = len(cleaned_data)
    gate_stats = {}

    for quad_name, quad_mask in quadrants.items():
        cells_in_quad = cleaned_data[quad_mask]
        percentage = (len(cells_in_quad) / total_cells) * 100
        gate_stats[f"{quad_name}_Percentage"] = percentage

        # Calculate GM and median for non-FSC/SSC channels
        for param in [x_param, y_param]:
            if not any(dim in param for dim in ["FSC", "SSC"]):
                gate_stats[f"{quad_name}_{param}_GM"] = (
                    gmean(cells_in_quad[param]) if len(cells_in_quad) > 0 else 0
                )
                gate_stats[f"{quad_name}_{param}_Median"] = (
                    cells_in_quad[param].median() if len(cells_in_quad) > 0 else 0
                )

    # Define positions for labels
    label_positions = {
        "Q1": (0.95, 0.95),
        "Q2": (0.05, 0.95),
        "Q3": (0.05, 0.05),
        "Q4": (0.95, 0.05),
    }

    # Add labels to corners
    for quad_name, position in label_positions.items():
        percentage = gate_stats[f"{quad_name}_Percentage"]
        ax.text(
            position[0],
            position[1],
            f"{quad_name}\n{percentage:.1f}%",
            horizontalalignment=(
                "right" if "Q1" in quad_name or "Q4" in quad_name else "left"
            ),
            verticalalignment=(
                "top" if "Q1" in quad_name or "Q2" in quad_name else "bottom"
            ),
            transform=ax.transAxes,
            fontsize=6,
            fontweight="bold",
            color="red",
        )
        

    # Calculate global GM and median for non-FSC/SSC channels
    for param in [x_param, y_param]:
        if not any(dim in param for dim in ["FSC", "SSC"]):
            gate_stats[f"Global_{param}_GM"] = gmean(cleaned_data[param])
            gate_stats[f"Global_{param}_Median"] = cleaned_data[param].median()

    # Add quadrant lines
    ax.axvline(x_mid, color="red", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="red", linestyle="--", linewidth=1)

    print(f"Finished plotting {scatter_type} scatter with manual gates for {file_name}")
    return gate_stats


def plot_96well_grid(
    data, metric, plot_title, well_plots_dir, parameter_name=None, vmin=None, vmax=None
):
    print(f"Starting plot_96well_grid for metric: {metric}")
    fig, ax = plt.subplots(figsize=(16, 9))

    # Define the full 96-well plate layout
    rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    columns = range(1, 13)

    # Ensure 'Well' column exists in the data
    if "Well" not in data.columns:
        print(
            f"Error: 'Well' column not found in data for metric {metric}. Skipping plot."
        )
        plt.close(fig)
        return

    # Extract row and column from Well
    data["Row"] = data["Well"].str[0]
    data["Column"] = data["Well"].str[1:].astype(int)

    # Determine color scale range
    valid_values = data[metric][np.isfinite(data[metric])]
    if len(valid_values) == 0:
        print(f"Error: No valid data for metric {metric}. Skipping plot.")
        plt.close(fig)
        return

    if vmin is None and vmax is None:
        vmin = 0 if "Percentage" in metric else valid_values.min()
        vmax = 100 if "Percentage" in metric else valid_values.max()

    print(f"Color scale range: {vmin} to {vmax}")

    # Ensure vmin and vmax are not equal
    if vmin == vmax:
        vmax = vmin + 1

    # Create a dictionary for quick data lookup
    data_dict = {
        (row, col): data[(data["Row"] == row) & (data["Column"] == col)][metric].values[
            0
        ]
        for row in rows
        for col in columns
        if not data[(data["Row"] == row) & (data["Column"] == col)].empty
    }

    # Plot each well
    for row_idx, row in enumerate(rows):
        for col in columns:
            value = data_dict.get((row, col), np.nan)
            if np.isfinite(value):
                color_val = (value - vmin) / (vmax - vmin)
                circle = plt.Circle(
                    (col - 1, 7 - row_idx),
                    0.4,
                    fill=True,
                    color=plt.cm.viridis(color_val),
                )
                ax.add_artist(circle)

                # Determine text color based on background brightness
                bg_color = plt.cm.viridis(color_val)
                text_color = (
                    "white" if mcolors.rgb_to_hsv(bg_color[:3])[2] < 0.6 else "black"
                )

                ax.text(
                    col - 1,
                    7 - row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                # Draw an empty circle for missing data
                circle = plt.Circle(
                    (col - 1, 7 - row_idx), 0.4, fill=False, color="gray"
                )
                ax.add_artist(circle)
                ax.text(
                    col - 1,
                    7 - row_idx,
                    "No Data",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    # Set the limits and remove axes
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Add row labels
    for i, row in enumerate(rows):
        ax.text(
            -0.8, 7 - i, row, ha="center", va="center", fontsize=12, fontweight="bold"
        )

    # Add column labels closer to the samples
    for i in columns:
        ax.text(
            i - 1, 7.6, str(i), ha="center", va="center", fontsize=12, fontweight="bold"
        )

    # Add title at the top
    plt.title(plot_title, fontsize=16, pad=20)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", aspect=30, pad=0.08)
    cbar.set_label(metric, fontsize=12)

    # Save the plot
    if parameter_name:
        plot_filename = (
            f"{parameter_name}_{metric.replace(' ', '_').lower()}_96well_grid.png"
        )
    else:
        plot_filename = f"{metric.replace(' ', '_').lower()}_96well_grid.png"
    plot_path = os.path.join(well_plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved 96-well grid plot for {metric} to: {plot_path}")


def calculate_triplicate_stats(df, metric_column, well_column="Well"):
    """
    Calculate mean and standard deviation for triplicates, handling partial plates.
    """
    df["Group"] = df[well_column].apply(lambda x: f"{x[0]}{(int(x[1:]) - 1) // 3 + 1}")
    grouped = df.groupby("Group")

    triplicate_stats = grouped.agg(
        {
            metric_column: [
                "mean",
                "std",
                "count",
            ]  # Add count to check for incomplete triplicates
        }
    ).reset_index()

    triplicate_stats.columns = [
        "Group",
        f"{metric_column}_Mean",
        f"{metric_column}_Std",
        f"{metric_column}_Count",
    ]

    # Filter out incomplete triplicates (you can adjust this if you want to include duplicates)
    triplicate_stats = triplicate_stats[triplicate_stats[f"{metric_column}_Count"] == 3]

    return triplicate_stats


def plot_triplicate_stats(
    triplicate_stats, metric_column, output_dir, title=None, custom_names=None
):
    if len(triplicate_stats) == 0:
        print(f"No complete triplicates found for {metric_column}. Skipping plot.")
        return

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define a custom color palette with 4 colors from inferno colormap
    # custom_palette = ['#000004', '#851255', '#f6715b', '#fcfdbf']
    # Define a custom color palette with 4 colors from viridis colormap
    # custom_palette = ['#440154', '#40478b', '#29788e', '#fde725']
    # Define a custom color palette with 4 colors from coolwarm colormap
    custom_palette = ["#3a4cc0", "#90afdb", "#f1a38b", "#b40326"]

    fig, ax = plt.subplots(figsize=(16, 8))  # Increased figure size

    groups = triplicate_stats["Group"]
    means = triplicate_stats[f"{metric_column}_Mean"]
    stds = triplicate_stats[f"{metric_column}_Std"]

    # Calculate positions for grouped bars
    group_size = 4
    bar_width = 0.6
    group_width = group_size * bar_width + 1.2  # Extra space between groups
    positions = [
        i * group_width + j * bar_width
        for i in range(len(groups) // group_size + 1)
        for j in range(group_size)
    ][: len(groups)]

    # Create a color list that repeats the custom palette
    colors = [custom_palette[i % len(custom_palette)] for i in range(len(groups))]

    # Plot bars
    bars = ax.bar(
        positions,
        means,
        width=bar_width,
        align="center",
        alpha=0.8,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Plot error bars (only positive)
    ax.errorbar(
        positions,
        means,
        yerr=[np.zeros_like(stds), stds],
        fmt="none",
        color="black",
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
    )

    # Customize the plot
    ax.set_xlabel("Sample Group", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_column, fontsize=12, fontweight="bold")
    ax.set_title(
        title or f"{metric_column} by Sample Group",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-ticks and labels
    if custom_names and len(custom_names) == len(groups):
        ax.set_xticks(positions)
        ax.set_xticklabels(custom_names, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=10)

    # Customize grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Calculate the maximum height (including error bars)
    max_height = max(mean + std for mean, std in zip(means, stds))
    label_height_1 = (
        max_height * 1.05
    )  # Place first line of labels 5% above the highest point
    label_height_2 = (
        max_height * 1.10
    )  # Place second line of labels 10% above the highest point

    # Add value labels above each bar, alternating between two lines
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i % 2 == 0:
            height = label_height_1
            va = "bottom"
        else:
            height = label_height_2
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{mean:.2f}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )

    # Adjust y-axis limit to accommodate labels
    ax.set_ylim(0, label_height_2 * 1.1)  # Add 10% padding above the highest labels

    # Add row labels and horizontal lines
    for i in range(0, len(groups), group_size):
        group_start = positions[i]
        group_end = positions[min(i + group_size - 1, len(positions) - 1)]
        group_center = (group_start + group_end) / 2

        # Add horizontal line
        ax.axhline(
            y=-0.05 * max_height,
            xmin=(group_start - bar_width / 2) / ax.get_xlim()[1],
            xmax=(group_end + bar_width / 2) / ax.get_xlim()[1],
            color="black",
            linewidth=1,
        )

        # Add row label
        row_letter = chr(65 + i // group_size)  # 65 is ASCII for 'A'
        ax.text(
            group_center,
            -0.1 * max_height,
            f"Row {row_letter}",
            ha="center",
            va="top",
            fontweight="bold",
        )

    # Adjust bottom margin to make room for row labels
    plt.subplots_adjust(bottom=0.2)

    # Customize spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_visible(False)  # Hide bottom spine

    # Add a legend for color groups
    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=f"Group {i+1}")
        for i, color in enumerate(custom_palette)
    ]
    ax.legend(
        handles=legend_elements,
        title="Color Groups",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=8,
    )

    plt.tight_layout()
    plot_filename = f"{title.replace(' ', '_').lower()}_triplicate_stats.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved triplicate plot for {metric_column} to: {plot_path}")

def process_fcs_files(directory, plots_config, results_directory):
    start_time = time.time()

    # Create output structure
    (
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    ) = create_output_structure(results_directory)

    scatter_dfs = {}
    histogram_dfs = {}
    singlet_stats = []

    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".fcs")
    ]
    if not files:
        print("No FCS files found in the directory.")
        return pd.DataFrame(), 0  # Return empty DataFrame and 0 runtime

    print(f"Found {len(files)} FCS files in {directory}")

    # Sort files using extract_well_key
    files.sort(key=lambda f: extract_well_key(os.path.basename(f))[1])

    # Define the number of rows and columns for the plots
    num_rows = 8
    num_cols = 12

    # Create figure for singlet gate plots
    fig_singlets, axes_singlets = plt.subplots(
        num_rows + 1,
        num_cols + 1,
        figsize=(42, 28),  # Adjusted figure size to 12:8 ratio
        gridspec_kw={
            "height_ratios": [0.5] + [1] * num_rows,
            "width_ratios": [0.5] + [1] * num_cols,
        },
    )
    fig_singlets.suptitle("Singlet Gates", fontsize=24, fontweight="bold", y=0.98)

    # Adjust the spacing between subplots
    fig_singlets.subplots_adjust(hspace=0.1, wspace=0.4)  # Adjusted hspace and wspace

    # Create figures for each plot configuration
    figs = {}
    axes = {}
    for config in plots_config.values():
        plot_key = f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
        fig, ax = plt.subplots(
            num_rows + 1,
            num_cols + 1,
            figsize=(42, 28),  # Adjusted figure size to 12:8 ratio
            gridspec_kw={
                "height_ratios": [0.5] + [1] * num_rows,
                "width_ratios": [0.5] + [1] * num_cols,
            },
        )
        fig.suptitle(
            f"{config['type'].capitalize()} Plot: {config['x_param']} vs {config.get('y_param', '')}",
            fontsize=24,
            fontweight="bold",
            y=0.95,
        )  # Adjusted y to reduce space

        # Adjust the spacing between subplots
        if config["type"] == "scatter":
            fig.subplots_adjust(
                hspace=0.6, wspace=1
            )  # Reduced hspace for scatter plots
        else:
            fig.subplots_adjust(hspace=0.6, wspace=0.5)  # Reduced hspace for histograms

        figs[plot_key] = fig
        axes[plot_key] = ax

    # Add column and row labels for all plots
    for plot_key, fig in figs.items():
        ax = axes[plot_key]
        for col in range(1, num_cols + 1):
            ax[0, col].text(
                0.5,
                0.2,
                str(col),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=18,
            )  # Increased font size and boldness
            ax[0, col].axis("off")

        for row in range(1, num_rows + 1):
            ax[row, 0].text(
                0.5,
                0.5,
                chr(64 + row),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=18,
            )  # Increased font size and boldness
            ax[row, 0].axis("off")

        # Remove the top-left empty cell
        ax[0, 0].axis("off")

        # For scatter plots, adjust the position of subplots to bring rows closer
        if "scatter" in plot_key:
            for row in range(
                1, num_rows + 1
            ):  # Start from the first row of actual plots
                for col in range(1, num_cols + 1):
                    pos = ax[row, col].get_position()
                    pos.y0 = pos.y0 + 0.02 * (
                        row - 1
                    )  # Move up by 2% of figure height for each row
                    pos.y1 = pos.y1 + 0.02 * (row - 1)
                    ax[row, col].set_position(pos)
                    ax[row, col].set_aspect(
                        "equal", adjustable="box"
                    )  # Make each subplot square

    # Ensure tight layout to minimize blank space
    for fig in figs.values():
        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust rect to leave less space for suptitle

    # Initialize a set to keep track of wells with data
    wells_with_data = set()

    # Calculate total number of iterations
    total_iterations = len(files) * (len(plots_config) + 1)  # +1 for singlet plots

    # Create a progress bar
    with tqdm(
        total=total_iterations, desc="Processing files", unit="plot", file=sys.stdout
    ) as pbar:
        for file in files:
            well_key, (row_letter, col_number) = extract_well_key(file)
            row = (
                ord(row_letter) - 64
            )  # Convert letter to row number (A=1, B=2, ..., H=8)
            col = col_number

            if row > num_rows or col > num_cols:
                print(
                    f"Skipping file {file} with well key {well_key} as it is out of the {num_rows}x{num_cols} grid."
                )
                continue

            wells_with_data.add((row, col))

            try:
                data, _ = read_fcs(file)
                if data is None:
                    pbar.update(1)
                    continue

                # Remove doublets
                (
                    singlets,
                    singlet_percentage,
                    total_events,
                    singlet_events,
                ) = remove_doublets(data)
                print(
                    f"File: {file}, Singlet percentage: {singlet_percentage:.2f}%, "
                    f"Total events: {total_events}, Singlet events: {singlet_events}"
                )
                singlet_stats.append(
                    {
                        "Well": well_key,
                        "Singlet_Percentage": singlet_percentage,
                        "Total_Events": total_events,
                        "Singlet_Events": singlet_events,
                    }
                )

                # Plot singlet gate
                ax_singlet = axes_singlets[row, col]
                plot_singlet_gate(data, ax=ax_singlet, file_name=well_key)

                for config in plots_config.values():
                    plot_key = f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
                    if config["type"] == "scatter":
                        # Process scatter plot
                        ax = axes[plot_key][row, col]
                        gate_stats = plot_scatter_with_manual_gates(
                            singlets,
                            config["x_param"],
                            config["y_param"],
                            well_key,
                            ax,
                            scatter_type=config.get("scatter_type", "scatter"),
                            cmap=config.get("cmap", "viridis"),
                            x_scale=config.get("x_scale", "linear"),
                            y_scale=config.get("y_scale", "linear"),
                            xlim=config.get("xlim"),
                            ylim=config.get("ylim"),
                            gridsize=config.get("gridsize", 100),
                            quadrant_gates=config.get("quadrant_gates"),
                        )

                        if gate_stats is not None:
                            gate_stats["Well"] = well_key
                            new_df = pd.DataFrame([gate_stats])
                            if plot_key not in scatter_dfs:
                                scatter_dfs[plot_key] = []
                            scatter_dfs[plot_key].append(new_df)

                    elif config["type"] == "histogram":
                        # Process histogram plot
                        ax = axes[plot_key][row, col]
                        stats = plot_histogram(
                            singlets,
                            config["x_param"],
                            well_key,
                            ax,
                            x_scale=config.get("x_scale", "linear"),
                            kde=config.get("kde", False),
                            color=config.get("color", "blue"),
                            xlim=config.get("xlim"),
                            gates=config.get("gates"),
                        )

                        if stats is not None:
                            stats["Well"] = well_key
                            new_df = pd.DataFrame([stats])
                            if plot_key not in histogram_dfs:
                                histogram_dfs[plot_key] = []
                            histogram_dfs[plot_key].append(new_df)

                pbar.update(1)  # Update progress bar

            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        # Fill empty wells with "No Data"
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                if (row, col) not in wells_with_data:
                    # Fill singlet gate plot
                    ax_singlet = axes_singlets[row, col]
                    ax_singlet.text(0.5, 0.5, "No Data", ha="center", va="center")
                    ax_singlet.axis("off")

                    # Fill other plots
                    for ax in axes.values():
                        ax[row, col].text(0.5, 0.5, "No Data", ha="center", va="center")
                        ax[row, col].axis("off")

        # Concatenate DataFrames and save statistics to CSV files
        for plot_key, df_list in scatter_dfs.items():
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                scatter_csv_path = os.path.join(
                    stats_dir,
                    f"{plot_key}_statistics_{os.path.basename(results_directory)}.csv",
                )
                df.to_csv(scatter_csv_path, index=False)
                print(f"Saved {plot_key} statistics to: {scatter_csv_path}")
                scatter_dfs[plot_key] = df  # Replace list with concatenated DataFrame

        for plot_key, df_list in histogram_dfs.items():
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                histogram_csv_path = os.path.join(
                    stats_dir,
                    f"{plot_key}_statistics_{os.path.basename(results_directory)}.csv",
                )
                df.to_csv(histogram_csv_path, index=False)
                print(f"Saved {plot_key} statistics to: {histogram_csv_path}")
                histogram_dfs[plot_key] = df  # Replace list with concatenated DataFrame

        # Save singlet statistics
        singlet_df = pd.DataFrame(singlet_stats)
        singlet_csv_path = os.path.join(
            stats_dir, f"singlet_statistics_{os.path.basename(results_directory)}.csv"
        )
        singlet_df.to_csv(singlet_csv_path, index=False)
        print(f"Saved singlet statistics to: {singlet_csv_path}")

        # Process triplicate plots based on configuration
        for config in plots_config.values():
            plot_key = (
                f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
            )
            if "triplicate_plots" in config:
                df_to_use = (
                    scatter_dfs[plot_key]
                    if config["type"] == "scatter"
                    else histogram_dfs[plot_key]
                )
                for plot_spec in config["triplicate_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    if metric in df_to_use.columns:
                        triplicate_stats = calculate_triplicate_stats(df_to_use, metric)

                        if not triplicate_stats.empty:
                            # Save triplicate statistics
                            triplicate_stats_filename = (
                                f"{plot_key}_{metric}_triplicate_statistics.csv"
                            )
                            triplicate_stats_path = os.path.join(
                                triplicate_stats_dir, triplicate_stats_filename
                            )
                            triplicate_stats.to_csv(triplicate_stats_path, index=False)
                            print(
                                f"Saved triplicate statistics for {metric} to: {triplicate_stats_path}"
                            )

                            # Plot triplicate statistics
                            plot_triplicate_stats(
                                triplicate_stats,
                                metric,
                                triplicate_plots_dir,
                                title=f"{plot_key}_{title}",
                            )
                        else:
                            print(
                                f"No complete triplicates found for {metric} in {plot_key}. Skipping plot and statistics."
                            )
                    else:
                        print(
                            f"Metric {metric} not found in dataframe for {plot_key}. Skipping triplicate plot."
                        )

        # Save updated dataframes
        for plot_key, df in scatter_dfs.items():
            csv_filename = f"{plot_key}_statistics_with_triplicates.csv"
            csv_path = os.path.join(stats_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Saved {plot_key} statistics with triplicates to: {csv_path}")

        for plot_key, df in histogram_dfs.items():
            csv_filename = f"{plot_key}_statistics_with_triplicates.csv"
            csv_path = os.path.join(stats_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Saved {plot_key} statistics with triplicates to: {csv_path}")

        # Save all main plots
        for plot_key, fig in figs.items():
            plot_filename = f"{plot_key}_plot_{os.path.basename(results_directory)}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {plot_key} plot: {plot_path}")

        # Save singlet gate plot
        singlet_plot_filename = (
            f"singlet_gates_plot_{os.path.basename(results_directory)}.png"
        )
        singlet_plot_path = os.path.join(plots_dir, singlet_plot_filename)
        fig_singlets.savefig(singlet_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig_singlets)
        print(f"Saved singlet gates plot: {singlet_plot_path}")

        # Generate and save 96-well plots
        for config in plots_config.values():
            plot_key = (
                f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
            )
            if "96well_plots" in config:
                df = (
                    scatter_dfs[plot_key]
                    if config["type"] == "scatter"
                    else histogram_dfs[plot_key]
                )
                for plot_spec in config["96well_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    if metric in df.columns:
                        parameter_name = config["x_param"].split("-")[
                            0
                        ]  # Extract parameter name (e.g., 'BL1' from 'BL1-H')
                        plot_96well_grid(
                            df,
                            metric,
                            title,
                            well_plots_dir,
                            parameter_name=parameter_name,
                        )
                    else:
                        print(
                            f"Warning: Metric {metric} not found in data for {plot_key}. Skipping 96-well plot."
                        )

        return scatter_dfs, histogram_dfs, singlet_df, time.time() - start_time
