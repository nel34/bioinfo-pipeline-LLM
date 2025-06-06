import os
from argparse import Namespace

import pytest

from fltower.run_args import parse_run_args


@pytest.mark.parametrize(
    "cli_args, expected",
    [
        # Test default arguments
        ([], Namespace(input=os.getcwd(), output=os.getcwd(), parameters=None)),
        # Test custom input and output directories
        (
            ["-I", "/path/to/input", "-O", "/path/to/output"],
            Namespace(
                input="/path/to/input", output="/path/to/output", parameters=None
            ),
        ),
        # Test custom parameters file
        (
            ["-P", "/path/to/params.json"],
            Namespace(
                input=os.getcwd(), output=os.getcwd(), parameters="/path/to/params.json"
            ),
        ),
        # Test all custom arguments
        (
            ["-I", "/input", "-O", "/output", "-P", "/params.json"],
            Namespace(input="/input", output="/output", parameters="/params.json"),
        ),
    ],
)
def test_parse_run_args(monkeypatch, cli_args, expected):
    """Test the parse_run_args function with different command-line arguments."""
    monkeypatch.setattr("sys.argv", ["fltower"] + cli_args)
    args = parse_run_args()
    assert args == expected


def test_parse_run_args_error(monkeypatch):
    """Test parse_run_args function with invalid arguments."""
    # Invalid argument
    monkeypatch.setattr("sys.argv", ["fltower", "--unknown"])
    with pytest.raises(SystemExit) as excinfo:
        parse_run_args()
    # Assert that the program exits with a status code of 2 (argparse error)
    assert excinfo.value.code == 2
