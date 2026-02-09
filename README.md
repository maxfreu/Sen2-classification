# Sentinel-2 Time Series Classification of Forest Tree Species

This repository accompanies our (yet to be published) paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5705254). What the code here basically does, is to train a GRU on [this dataset](https://www.openagrar.de/receive/openagrar_mods_00094435) (description [here](https://essd.copernicus.org/articles/17/351/2025/essd-17-351-2025.html)). The resulting maps and additional data can be found [here](https://data.goettingen-research-online.de/dataverse/tsmg).

## Training your own model on the dataset

The published models come from the output of `experiments/final_crossvalidation.py`. But if you want to train your own models, you can probably do that easiest by using `main_CLI.py` and combining the different config files for the data and model.

`python main_CLI.py --config configs/gru.yaml --config configs/13_classes.yaml --config configs/statistics_223_g-5k.yaml`

I haven't touched the code in a while, so don't hesitate to raise issues when you encounter them.

## Inference

The inference for the published maps has been run on a HPC using slurm and the `inference_pipeline.py` script, submitted via `inference_germany_pipeline.sh`. It assumes that there are four GPUs per node. It basically works through the tiles in a FORCE datacube in an embarassingly parallel manner; each GPU processes a folder. The core inference logic is implemented in `sen2classification/models/satellite_classifier.py - 'predict_force_folder'`.
