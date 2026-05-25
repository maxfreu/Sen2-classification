import os
import yaml
from experiments.train_and_validate import load_data, train_and_validate


logdir = "output"
experiment_name = "input_band_ablation"
model_config = "/home/max/dr/Sen2-classification/configs/gru_best.yaml"
band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B11", "B12"]


with open("configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

runs = [None, *range(10)]
run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
omitted_band_index = runs[run_id]

input_band_indices = None if omitted_band_index is None else [idx for idx in range(10) if idx != omitted_band_index]
satellite_input_channels = 10 if omitted_band_index is None else 9
omitted_band = "none" if omitted_band_index is None else band_names[omitted_band_index]
version = "baseline_all_bands" if omitted_band_index is None else f"without_{omitted_band.lower()}"

data, dataconfig = load_data(
    overwrite_args={
        "mean": norm_config["mean"],
        "stddev": norm_config["stddev"],
        "input_band_indices": input_band_indices,
        "satellite_input_channels": satellite_input_channels,
    }
)

train_and_validate(
    model_config,
    data,
    dataconfig | {"normalization": "223_g-5k", "omitted_band": omitted_band},
    logdir,
    experiment_name=experiment_name,
    version=version,
    experiment_file=__file__,
    model_extra_args={
        "num_classes": data.num_classes,
        "classes": data.classes,
        "satellite_input_channels": satellite_input_channels,
    },
)