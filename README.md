# bioinfo-pipeline-LLM with FL tower code

![Pytest](https://github.com/synthetic-biology-group-cbs-montpellier/FLtower/actions/workflows/test.yml/badge.svg?branch=main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## 1. Installation

|OS|Linux|Windows|Mac|
|:-:|:-:|:-:|:-:|
|**compatibility**|Yes|Yes|Yes|

### [Optional] Install conda

*We use conda environment to avoid version problem between FLtower dependencies and other applications.*

We recommend to download the lighter version via [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) (if you only intend to use FLtower without developing new feature).

### [Optional] Create the environment
```bash
conda create -n fltower python=3.11
conda activate fltower
```

---

### Install FLtower from PyPi
   ```bash
   pip install fltower
   ```

To check if FLtower is well installed, run:
```bash
fltower --help
```


## **2. Usage**

### Folder structure

Your input data folder should look like this:

```
my_root/
├── my_data/
│   ├── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_A1.fcs
│   ├── ...
│   ├── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_H12.fcs
│   └── parameters.json
```
With 96 `FCS` files and one `parameters.json` file.

### Parameter file

The parameter file brings together all the plot configurations.

You can find a [template here](https://raw.githubusercontent.com/synthetic-biology-group-cbs-montpellier/FLtower/refs/heads/main/fltower/resource/parameters.json), or just run `fltower` inside a folder without any `parameters.json` file and the software will generate for you a `parameters_template.json`.

### Running FLtower *(5~10 min)*

*If you use it, remember to activate your environment* (*`conda activate fltower`*),

#### Basic run

You can run FLtower from your data folder (in this example: *`cd my_root/my_data/`*) with just this keyword:

```bash
fltower
```

And the software will generate a `results_<DATE>_<TIME>` folder inside your input folder like this:

```
my_root/
└── my_data/
    ├── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_A1.fcs
    ├── ...
    ├── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_H12.fcs
    ├── parameters.json
    └── results_20241127_094857/
        ├── plots/
        ├── .../
        ├── statistics/
        ├── summary_report.pdf
        └── used_parameters.json
```

#### [Optional] Custom run

You can also run FLtower from anywhere by specifying your input data folder, where you want to save outputs and where are located the parameter file:

```bash
fltower --input my_root/my_data/ --output my_root/OUTPUT/ --parameters my_root/parameters.json
```
Or with the shortcuts:

```bash
fltower -I my_root/my_data/ -O my_root/OUTPUT/ -P my_root/parameters.json
```
Then your folder structure should look like this:
```
my_root/
├── parameters.json
├── my_data/
│   ├── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_A1.fcs
│   ├── ...
│   └── RBS library - 7975 - Caff_Experiment_M9 - 7975 - P1 - Caff_H12.fcs
└── OUTPUT/
    └── results_20241127_094857/
        ├── plots/
        ├── .../
        ├── statistics/
        ├── summary_report.pdf
        └── used_parameters.json
```

---

## 3. **Support**

If you encounter any issues or have questions, [open an issue in the GitHub repository](https://github.com/synthetic-biology-group-cbs-montpellier/FLtower/issues).
