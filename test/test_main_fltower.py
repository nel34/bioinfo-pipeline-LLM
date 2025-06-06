import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from fltower.main_fltower import main

# Get the directory of the current test file
TEST_DIR = os.path.dirname(__file__)

# Paths relative to the test directory
REFERENCE_INPUT_DIR = os.path.join(TEST_DIR, "reference_input")
REFERENCE_OUTPUT_DIR = os.path.join(TEST_DIR, "reference_output")


def compare_csv_files(file1, file2):
    """Compare two CSV files for equality."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure columns and rows match
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def get_all_filepaths(folder_path):
    """
    Get the file paths of all files in a folder, including subfolders.

    Parameters
    ----------
    folder_path : str
        Path to the folder to scan.

    Returns
    -------
    list of str
        A list of file paths for all files in the folder.
    """
    filepaths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths


def associate_files(generated_csv, reference_csv):
    """
    Associate files between generated and reference directories.

    Parameters
    ----------
    generated_csv : list of str
        List of file paths from the generated directory.
    reference_csv : list of str
        List of file paths from the reference directory.

    Returns
    -------
    common_files : list of tuple
        List of tuples (generated_file, reference_file) for files that match.
    left_only : list of str
        List of files present only in `generated_csv`.
    right_only : list of str
        List of files present only in `reference_csv`.
    """

    # Normalize file paths to compare relative paths
    def _normalize(filepath):
        filebase = os.path.basename(filepath)
        if "_results_" in filebase:
            filebase = filebase.split("_results_")[0] + ".csv"
        return filebase

    # Build dictionaries for quick lookup by normalized paths
    generated_map = {_normalize(f): f for f in generated_csv}
    reference_map = {_normalize(f): f for f in reference_csv}

    # Identify matches, left-only, and right-only
    common_keys = set(generated_map.keys()) & set(reference_map.keys())
    left_only_keys = set(generated_map.keys()) - set(reference_map.keys())
    right_only_keys = set(reference_map.keys()) - set(generated_map.keys())

    # Build the result lists
    common_files = [(generated_map[key], reference_map[key]) for key in common_keys]
    left_only = [generated_map[key] for key in left_only_keys]
    right_only = [reference_map[key] for key in right_only_keys]

    return common_files, left_only, right_only


def compare_directories(generated_dir, reference_dir):
    """Compare generated CSV files with reference directory."""

    generated_files = get_all_filepaths(generated_dir)
    reference_files = get_all_filepaths(reference_dir)

    # Helper function to filter files based on excluded extensions
    def _filter_files(files):
        # Define a set of excluded extensions
        excluded_extensions = {".png", ".json", ".pdf"}
        return [
            f for f in files if not any(f.endswith(ext) for ext in excluded_extensions)
        ]

    # Filter files to exclude PNG, JSON and PDF
    generated_csv = _filter_files(generated_files)
    reference_csv = _filter_files(reference_files)

    common_files, left_only, right_only = associate_files(generated_csv, reference_csv)

    # Check for any mismatches in files or subdirectories
    assert not left_only, f"Extra files in generated output: {left_only}"
    assert not right_only, f"Missing files in generated output: {right_only}"

    # Compare common files
    for generated_file, reference_file in common_files:
        compare_csv_files(generated_file, reference_file)


@pytest.fixture
def temporary_output_dir():
    """Create a temporary directory for script outputs."""
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_normal_pipeline(temporary_output_dir):
    """Test non-regression by comparing script outputs to reference outputs."""
    main(["-I", REFERENCE_INPUT_DIR, "-O", temporary_output_dir])
    # Compare the generated output to the reference output
    compare_directories(temporary_output_dir, REFERENCE_OUTPUT_DIR)


def test_run_without_parameters_file(temporary_output_dir):
    """Test helping feature when user doesn't provide any params file."""
    main(["-I", temporary_output_dir, "-O", temporary_output_dir])
    # Verify the file exists
    assert os.path.exists(
        os.path.join(temporary_output_dir, "parameters_template.json")
    )
