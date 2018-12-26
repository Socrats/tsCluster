# tsCluster
Clustering time-series

The objective of this repository is to provide efficient and easily accessible SoA methods for clustering time-series.
Also, we want to provide tests reproducing the results of the papers in which these methods were introduced and a common
benchmark for of them.


## Requirements
This repository requires Cython and clang for compiling.

To compile run `python setup.py build_ext --inplace`

For the moment you also need Numpy. In future version we will
replace the numpy dependencies with our own c++ wrappers.