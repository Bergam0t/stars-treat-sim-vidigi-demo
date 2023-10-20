[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythonhealthdatascience/stars-treat-sim/HEAD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10026327.svg)](https://doi.org/10.5281/zenodo.10026327)
[![PyPI version fury.io](https://badge.fury.io/py/treat-sim.svg)](https://pypi.org/project/treat-sim/)

[<img src="https://img.shields.io/static/v1?label=dockerhub&message=images&color=important?style=for-the-badge&logo=docker">](https://hub.docker.com/r/tommonks01/treat_sim)


# Towards Sharing Tools, and Artifacts, for Reusable Simulation: a minimal model examplar

## Overview

The materials and methods in this repository support work towards developing the S.T.A.R.S healthcare framework (**S**haring **T**ools and **A**rtifacts for **R**euable **S**imulations in healthcare).  The code and written materials here demonstrate the application of S.T.A.R.S' version 1 to sharing a `simpy` discrete-event simuilation model and associated research artifacts.  

* All artifacts in this repository are linked to study researchers via ORCIDs;
* Model code is made available under an MIT license;
* Python dependencies are managed through `conda`;`
* Documentation of the model is enhanced using a Jupyter notebook.
* The python code itself can be viewed and executed in Jupyter notebooks via [Binder](https://mybinder.org); 
* The materials are deposited and made citatable using Zenodo;
* The model is sharable with other researchers and the NHS without the need to install software.

## Author ORCIDs

[![ORCID: Harper](https://img.shields.io/badge/ORCID-0000--0001--5274--5037-brightgreen)](https://orcid.org/0000-0001-5274-5037)
[![ORCID: Monks](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)

## Funding

This code is part of independent research supported by the National Institute for Health Research Applied Research Collaboration South West Peninsula. The views expressed in this publication are those of the author(s) and not necessarily those of the National Institute for Health Research or the Department of Health and Social Care.

## Instructions to run the model

### Online Notebooks via Binder

The python code for the model has been setup to run online in Jupyter notebooks via binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythonhealthdatascience/stars-treat-sim/HEAD)

> mybinder.org is a free tier service.  If it has not been used in a while Binder will need to re-containerise the code repository, and push to binderhub. This will take several minutes. After that the online environment will be quick to load.

### To download code and run locally

#### Downloading the code

Either clone the repository using git or click on the green "code" button and select "Download Zip".

```bash
git clone https://github.com/pythonhealthdatascience/stars-treat-sim
```

#### Installing dependencies

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend install [mini-conda](https://docs.conda.io/en/latest/miniconda.html); navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

```
conda env create -f binder/environment.yml
```

Activate the conda environment using the following command

```
conda activate stars_treat_sim
```

#### Running the model

To run 50 multiple replications of across a number of example experiments use the following code


```python
from treat_sim.model import (get_scenarios, run_scenario_analysis,
                             scenario_summary_frame, 
                             DEFAULT_RESULTS_COLLECTION_PERIOD)

if __name__ == '__main__':

    results = run_scenario_analysis(get_scenarios(), 
                                    DEFAULT_RESULTS_COLLECTION_PERIOD,
                                    n_reps=50)

    results_summary = scenario_summary_frame(results)
    print(results_summary)

```

Alternative you can design and execute individual experiments by creating a `Scenario` object

```python
from treat_sim.model import Scenario, multiple_replications

if __name__ == '__main__':

    # use all default parameter values
    base_case = Scenario()

    results = multiple_replications(base_case).describe().round(2).T
    print(results)

```


## Repo overview

```
.
├── binder
│   └── environment.yml
├── LICENSE
├── MANIFEST.in
├── notebooks
│   └── test_package.ipynb
├── README.md
├── requirements.txt
├── setup.py
└── treat_sim
    ├── data
    │   └── ed_arrivals.csv
    ├── distributions.py
    ├── __init__.py
    └── model.py
```

* `binder` - contains the environment.yml file (sim) and all dependencies managed via conda
* `data` - directory containing data files used by analysis notebooks. 
* `LICENSE` - details of the MIT permissive license of this work.
* `notebooks` - contains a notebook to run the model and provides basic enhanced model documentation.
* `main.py` - an example simpy model to use to test the virtual environment 
* `README` - what you are reading now!
* `treat_sim` - contains a packaged version of the model.


## Citation

If you use the materials within this repository we would appreciate a citation.

```bibtex
@software{monks_2023_10026327,
  author       = {Monks, Thomas and
                  Harper, Alison},
  title        = {{Towards Sharing Tools, and Artifacts, for Reusable 
                   Simulation: a minimal model examplar}},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  url 	       = {https://zenodo.org/records/10026327}
}
```

