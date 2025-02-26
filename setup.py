#!/usr/bin/env python

from distutils.core import setup

setup(name='sen2classification',
      version='0.1.0',
      description='Sentinel 2 time series classification via GRU and BERT.',
      author='MF',
      packages=['sen2classification'],
      install_requires=[
          "numpy>=1.0.0",
          "pandas>=1.0.0",
          "duckdb>=1.0.0",
          "torch>=2.0.0",
          "pytorch-lightning>=2.0.0",
          "gdal>=3.0.0",
          "torchmetrics>=0.11.0",
          "torchvision>=0.15",
          "numba"
      ]
     )