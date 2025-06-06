import json
import os
import shutil


def save_parameters_template(input_dir):
    template_filepath = os.path.join(input_dir, "parameters_template.json")
    module_dir = os.path.dirname(__file__)
    source_filepath = os.path.join(module_dir, "resource", "parameters.json")
    return shutil.copyfile(source_filepath, template_filepath)


def load_parameters(input_dir, parameters_file=None):
    """
    Load plotting parameters from a JSON file.

    Parameters
    ----------
    input_dir : str
        Directory where the default parameters file is stored.
    parameters_file : str or None
        Path to the parameters JSON file. If None, defaults to `parameters.json` in `input_dir`.

    Returns
    -------
    dict
        The loaded parameters.
    """
    # Determine file path
    file_path = (
        parameters_file
        if parameters_file
        else os.path.join(input_dir, "parameters.json")
    )

    # Check if the file exists
    if not os.path.exists(file_path):
        dest_path = save_parameters_template(input_dir)
        raise FileNotFoundError(
            f"'parameters.json' file not found: {file_path}\nAn example is saved to help you: {dest_path}"
        )

    # Load the JSON data
    with open(file_path, "r") as f:
        return json.load(f)


def save_parameters(parameters, output_folder, file_name="parameters.json"):
    """
    Save parameters to a JSON file in the specified output folder.

    Parameters
    ----------
    parameters : dict
        The parameters to save.
    output_folder : str
        The directory where the file will be saved.
    file_name : str, optional
        The name of the file to save (default is "parameters.json").

    Returns
    -------
    str
        The path to the saved file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the full path to the file
    file_path = os.path.join(output_folder, file_name)

    # Save the parameters to the file
    with open(file_path, "w") as f:
        json.dump(parameters, f, indent=4)

    return file_path
