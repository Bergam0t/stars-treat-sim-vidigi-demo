# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates formatted as YYYY-MM-DD as per [ISO standard](https://www.iso.org/iso-8601-date-and-time-format.html).

Consistent identifier (represents all versions, resolves to latest): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10026326.svg)](https://doi.org/10.5281/zenodo.10026326)

## [v1.1.1] - 2024-05-01

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11098944.svg)](https://doi.org/10.5281/zenodo.11098944)

### Fixed

* Trauma patient treatment fixed to use correct distribution and parameters.

## [v1.1.0] - 2024-04-06

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10936052.svg)](https://doi.org/10.5281/zenodo.10936052)

### Changed

* Upgraded internal implementation of generating non-overlapping random number streams. This is now implemented to use `np.random.SeedSequence`. See [https://numpy.org/doc/stable/reference/random/parallel.html](https://numpy.org/doc/stable/reference/random/parallel.html).

### Fixed

* `setup.py` now links to correct Github URL

## [v1.0.0] - 2023-10-20

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10026327.svg)](https://doi.org/10.5281/zenodo.10026327)

:seedling: First release.

[1.1.1]: https://github.com/pythonhealthdatascience/stars-treat-sim/releases/tag/1.1.1

[1.1.0]: https://github.com/pythonhealthdatascience/stars-treat-sim/releases/tag/1.1.0

[1.0.0]: https://github.com/pythonhealthdatascience/stars-treat-sim/releases/tag/1.0.0