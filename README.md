# astra

The **astra** package (**A**stronomical **S**pectroscopy **T**ools for **R**apid **A**nalysis) is a small library of methods and tools for quick analysis of astronomical spectra, with a particular focus on spectra in time series.

astra is designed to be a simple toolbox of various methods used in spectral analysis, such as cross-correlation, optimal subtraction, and flux resampling. It is intended to be flexible, allowing underlying functions in top-level methods to be easily swapped for alternatives either provided within astra or defined by the user. It is also designed to be quick to get started with, by having minimal dependencies and proprietary objects and instead relying almost entirely on NumPy arrays to pass data around.

astra also intends to make older methods still used today easy to access and maintain, by providing native, well-documented, and Python-optimised versions that are simple to understand. The first methods ported as such come from the Fortran 77 program molly. 
