from setuptools import find_packages, setup

# package description
DESCRIPTION = "Software for cytometry analysis."

# long description of the package
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


def read_version():
    with open("fltower/__version__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


REQUIREMENTS = [
    "numpy>=1.26.4",
    "pandas>=2.2.3",
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
    "scipy==1.13.0",
    "fcsparser==0.2.8",
    "tqdm==4.67.1",
]

setup(
    name="fltower",
    version=read_version(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="GNU GPL v3",
    entry_points={
        "console_scripts": [
            "fltower=fltower.main_fltower:main",
        ],
    },
    install_requires=REQUIREMENTS,
    include_package_data=True,
    packages=find_packages(),
)
