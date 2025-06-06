# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser


def parse_run_args(command_line_arguments=None):
    """Parse run arguments

    Parameters
    ----------
    command_line_arguments : List[str]
        Used to give inputs for the runtime when you call this function like a module.
        For example, to test the FLtower run from tests folder.

    Returns
    -------
    ArgumentParser.args
        An accessor of run arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-I",
        "--input",
        type=str,
        default=os.getcwd(),
        help="Folder path with input data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        default=os.getcwd(),
        help="Folder path for output data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        default=None,
        help="Path of the parameters.json file.\nDEFAULT: FLtower expect this file inside your current directory",
    )

    return parser.parse_args(command_line_arguments)
