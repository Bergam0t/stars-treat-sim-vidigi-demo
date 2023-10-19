[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythonhealthdatascience/stars-treat-sim/HEAD)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6497477.svg)](https://doi.org/10.5281/zenodo.6497477)
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

## Online Notebooks via Binder

The python code for the model has been setup to run online in Jupyter notebooks via binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythonhealthdatascience/stars-treat-sim/HEAD)

> Binder is a free service.  If it has not been used in a while Binder will need to re-containerise the code repository, and push to binderhub. This will take several minutes. After that the online environment will be quick to load.

## Repo overview

```
.
├── binder
│   └── environment.yml
├── data
│   └── ed_arrivals.csv
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
