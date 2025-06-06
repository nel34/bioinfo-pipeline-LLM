import json
import os
from tempfile import TemporaryDirectory

import pytest

from fltower.data_manager import load_parameters, save_parameters


@pytest.fixture
def example_parameters():
    """Provide example JSON parameters data."""
    return {
        "plots_config_1": {
            "type": "scatter",
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "x_scale": "log",
            "y_scale": "log",
            "xlim": [1, 200000],
            "ylim": [1, 200000],
            "cmap": "inferno",
            "gridsize": 100,
            "scatter_type": "density",
            "quadrant_gates": {"x": 2600, "y": 2000},
            "96well_plots": [
                {"metric": "Q3_Percentage", "title": "OFF cells Percentage"},
                {"metric": "Q2_Percentage", "title": "RFP cells Percentage"},
                {"metric": "Q4_Percentage", "title": "GFP cells Percentage"},
                {
                    "metric": "Global_BL1-H_Median",
                    "title": "Scatter Plot-GFP-RFP:GFP Median",
                },
            ],
            "triplicate_plots": [
                {"metric": "Q3_Percentage", "title": "OFF cells Percentage"},
                {"metric": "Q2_Percentage", "title": "RFP cells Percentage"},
                {"metric": "Q4_Percentage", "title": "GFP cells Percentage"},
            ],
        },
        "plots_config_2": {
            "type": "histogram",
            "x_param": "BL1-H",
            "x_scale": "log",
            "xlim": [1, 200000],
            "color": "seagreen",
            "kde": True,
            "gates": [[10, 800], [800, 100000]],
            "96well_plots": [
                {"metric": "Global_Median", "title": "GFP Histogram-Median"}
            ],
            "triplicate_plots": [{"metric": "Global_Median", "title": "GFP-Median"}],
        },
        "plots_config_3": {
            "type": "histogram",
            "x_param": "YL2-H",
            "x_scale": "log",
            "xlim": [1, 200000],
            "color": "coral",
            "kde": True,
            "gates": [[10, 800], [800, 100000]],
            "96well_plots": [
                {"metric": "Global_Median", "title": "RFP Histogram-Median"}
            ],
            "triplicate_plots": [{"metric": "Global_Median", "title": "RFP-Median"}],
        },
    }


def test_load_parameters_with_explicit_file(example_parameters):
    """Test loading parameters with an explicit parameters file."""
    with TemporaryDirectory() as temp_dir:
        # Create a temporary parameters.json file
        parameters_path = os.path.join(temp_dir, "parameters.json")
        with open(parameters_path, "w") as f:
            json.dump(example_parameters, f)

        # Load parameters using the function
        params = load_parameters(input_dir=temp_dir, parameters_file=parameters_path)
        assert params == example_parameters


def test_load_parameters_with_default_file(example_parameters):
    """Test loading parameters from the default location."""
    with TemporaryDirectory() as temp_dir:
        # Create a default parameters.json file in the input directory
        parameters_path = os.path.join(temp_dir, "parameters.json")
        with open(parameters_path, "w") as f:
            json.dump(example_parameters, f)

        # Load parameters using the function without specifying the file
        params = load_parameters(input_dir=temp_dir)
        assert params == example_parameters


def test_load_parameters_file_not_found():
    """Test handling of missing parameters file."""
    with TemporaryDirectory() as temp_dir:
        # Ensure no parameters.json file exists
        non_existent_path = os.path.join(temp_dir, "parameters.json")
        # Escape backslashes or use a raw string for the regex match
        escaped_path = non_existent_path.replace("\\", "\\\\")  # Escape backslashes
        with pytest.raises(
            FileNotFoundError,
            match=rf"'parameters.json' file not found: {escaped_path}",
        ):
            # Expect a FileNotFoundError
            load_parameters(input_dir=temp_dir)


def test_load_parameters_invalid_json():
    """Test handling of invalid JSON content."""
    with TemporaryDirectory() as temp_dir:
        # Create an invalid parameters.json file
        parameters_path = os.path.join(temp_dir, "parameters.json")
        with open(parameters_path, "w") as f:
            f.write("INVALID JSON CONTENT")

        # Expect a JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            load_parameters(input_dir=temp_dir, parameters_file=parameters_path)


def test_save_parameters():
    """Test saving parameters to a file."""
    # Example parameters
    parameters = {
        "config_1": {"type": "scatter", "x_param": "BL1-H", "y_param": "YL2-H"},
        "config_2": {"type": "histogram", "x_param": "BL1-H"},
    }

    # Create a temporary directory for testing
    with TemporaryDirectory() as temp_dir:
        # Call the function
        saved_file = save_parameters(parameters, temp_dir)

        # Verify the file exists
        assert os.path.exists(saved_file)

        # Verify the contents of the file
        with open(saved_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data == parameters


def test_save_parameters_custom_file_name():
    """Test saving parameters with a custom file name."""
    parameters = {"key": "value"}
    custom_file_name = "custom_params.json"

    with TemporaryDirectory() as temp_dir:
        saved_file = save_parameters(parameters, temp_dir, file_name=custom_file_name)

        # Verify the custom file exists
        assert os.path.exists(saved_file)

        # Verify the file name
        assert os.path.basename(saved_file) == custom_file_name

        # Verify the contents of the file
        with open(saved_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data == parameters
