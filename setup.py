#!/usr/bin/env python

from distutils.core import setup

setup(name='TreeClassifier',
      version='0.1.0',
      description='ResNet18 used for tree classification',
      author='MF',
      packages=['treeclassifier'],
      install_requires=[
          "numpy>=1.0.0",
          "torch>=2.0.0",
          "pytorch-lightning>=2.0.0",
          "gdal>=3.0.0",
          "torchmetrics>=0.11.0",
          "torchvision>=0.15",
          "pandas>=1.0.0",
          "rioxarray"
      ]
     )