# astra

The **astra** package (**A**stronomical **S**pectroscopy **T**ools for **R**apid **A**nalysis) is a small library of methods and tools for quick analysis of astronomical spectra, with a particular focus on spectra in time series.

astra is designed to be a simple toolbox of various methods used in spectral analysis, such as cross-correlation, optimal subtraction, and flux resampling. It is intended to be flexible, and easy to use, by having minimal dependencies and/or proprietary objects -- most methods work simiply by using NumPy arrays as spectra inputs and outputs.

astra also intends to make some older methods still used today easy to access and maintain, by providing native, well-documented, and Python-optimised versions that are simple to understand. The first methods ported as such come from the Fortran 77 program [molly](https://cygnus.astro.warwick.ac.uk/phsaap/software/molly/html/USER_GUIDE.html) (e.g. `correlate.xcorrelate` and `fitting.optsub`).

## Installation

To install astra, you can simply download and install in one line with `pip`:

```
pip install https://github.com/jsimpson-astro/astra/archive/main.zip
```

## Overview of available methods

astra is organised into three main sub-modules and two auxiliary sub-modules:

* **correlate** - methods for correlation/cross-correlation. Provides `xcorrelate`, which is a flexible function for performing cross-correlation analysis between a set of observed and template spectra. Masking is supported.

* **fitting** - assorted fitting methods, including:
  - `optsub`: a simple function for performing optimal subtraction between a set of spectra and templates. Masking is supported.
  - `linefitmc`: a complex function for multiple-line fitting using `emcee`.
  - `rvmc`: a function for fitting radial velocities using `emcee`.
  - `SpectrumInterpolator`: a class for interpolating between model spectra of known parameters, and
  - `SpectrumFitter`: a class for fitting spectra to models using emcee and an instance of`SpectrumInterpolator`. Priors may be applied to fitting -- currently uniform and Gaussian priors are provided (by `UniformPrior` and `GaussianPrior`, respectively).
  The fitting submodule also provides many simple spectral models, such as blackbody models and power laws, under the `specmodels` sub-module.

* **integrate** - various methods for integrating spectra. Currently provides one operation, with more on the way:
  - `ew`: computes the equivalent width of one or multiple spectra. Masking is supported.

The two other sub-modules are **utils**, which contains various functions for performing simple operations on spectra (e.g. averaging, Doppler shifting), and **plotting**, which provides some general plotting methods for convenient display of results from astra methods.

The **molly** sub-module is deprecated, and will be removed in a future version. The structure may change in future versions, but deprecation warnings will be issued well in advance with old methods still accessible for backwards-compatability with old analysis scripts.

## Acknowledgement

If you find astra useful in your work, please reference this repository (https://github.com/jsimpson-astro/astra) in a footnote. If you encounter any problems, have any questions, or come up with any suggestions, please let me know!


