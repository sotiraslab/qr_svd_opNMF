# [Scalable NMF via linearly optimized data compression](https://doi.org/10.1117/12.2654282)

## Introduction

This is the accompanying python code for the 2023 SPIE Medical Imaging Conference manuscript "[Scalable NMF via linearly optimized data compression](https://doi.org/10.1117/12.2654282)". This implementation uses linear optimization using either QR or SVD to compress data and to reduce memory footprint and computational load of orthonormal projective non-negative matrix factorization (opNMF) introduced in [Linear and Nonlinear Projective Nonnegative Matrix Factorization](https://doi.org/10.1109/TNN.2010.2041361). The opNMF implementation for neuroimaging context is a stripped down port of the matlab `opnmf.m` and `opnmf_mem.m` codes found at [brainparts github repository](https://github.com/asotiras/brainparts) to python. 

## Prerequisites

This code was tested on Linux (__Rocky Linux release 8.8__) operating system using python installed through [anaconda](https://www.anaconda.com/). The following python packages, available either through the default repository or the conda-forge repository, are required to run this code.

| Package Name | Package Version Tested | Notes |
| :----------: | :--------------------: | :---- |
| numpy | 1.20.3 |  |
| scipy | 1.7.1 | |
| scikit-learn | 1.0.1 | |
| hdf5storage | 0.1.16 | this is used to save and load input, intermediate, and final output files in compressed hdf5 format that can also be loaded in MatLab. |

## Quick Start

1. You will need to prepare a nonnegative input data matrix in hdf5 compressed format with `.mat` extension, saved with variable name `X`.
1. For example, you can create a random data matrix of size of size 5000 by 1000, save it to a `input.mat` file, using the following python snippet.
    ```
    import numpy as np
    import hdf5storage

    X = np.random.rand(5000, 1000)
    hdf5storage.savemat("./input.mat", mdict = {"X": X})
    ```
1. Note that unlike [brainparts github repository](https://github.com/asotiras/brainparts) implementation that has additional preprocessing to remove common zero pixels/voxels across all columns and to downsample the `X` matrix prior to multiplicative updates, this implementation does NOT have such preprocessing. The appropriate preprocessing and downsampling is left to the end user to carry out prior to calling this code snippet.
1. Determine how many components you want to generate? We will call this value target rank. Default is $20$.
1. Determine how many iterations to run before terminating the multiplicative updates. Default is $1.0 \times 10^4$.
1. Determine what early stopping criterion threshold to use such that if $ {(\| {W}_{t+1} - {W}_{t} \|)}^{2}_{F} $ / $ {(\| W_{t+1} \|)}^{2}_{F} $ $ < tol $.
    * If the condition is met, the update will terminate. Default is $1.0 \times 10^{-6}$.
1. Determine how you want your initial component matrix $W$ to be intialized. Default is to use [nndsvd](https://doi.org/10.1016/j.patcog.2007.09.010).
1. Determine where the outputs will be generated.
1. If using QR, SVD, or truncated SVD opNMF, determine what rank of QR or SVD you will use to compress $X$ matrix.
1. Call the `Python/opnmf.py` script with appropriate parameters.
    1. for opNMF with speed optimized but high memory footprint resource usage:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="original" --tol=0.000001
    ```
    1. for opNMF with memory optimized but speed inefficient resource usage:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="mem" --tol=0.000001
    ```
    1. for q-opNMF (qr-opNMF) using qr rank of 128:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="qr" --tol=0.000001 --qrRank=128
    ```
    1. for s-opNMF (svd-opNMF) using SVD rank of 128:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="svd" --tol=0.000001 --svdRank=128
    ```
    1. for t-opNMF (truncated SVD-opNMF) using SVD rank of 128:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="truncsvd" --tol=0.000001 --svdRank=128 --n_iter=5 --n_oversamples=256
    ```
