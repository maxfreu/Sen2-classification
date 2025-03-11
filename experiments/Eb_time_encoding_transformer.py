import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "time_encoding"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]


for time_encoding in ("doy", "absolute"):
    if time_encoding == "doy":
        max_time = 366 + 1
    else:
        max_time = 8*366 + 1

    # for return_mode in ("single", "double", "random"):
    for return_mode in ("single", ):
        version = f"time_encoding={time_encoding}-return_mode={return_mode}"

        for model_config in ("configs/transformer.yaml",):
            data, dataconfig = load_data(overwrite_args={"time_encoding": time_encoding,
                                                    "return_mode": return_mode,
                                                    "mean": norm_config["mean"],
                                                    "stddev": norm_config["stddev"]})
            try:
                train_and_validate(model_config,
                                   data,
                                   dataconfig | {"normalization": "223_g-5k"},
                                   logdir,
                                   experiment_name=experiment_name,
                                   version=version,
                                   experiment_file=__file__,
                                   model_extra_args={"num_classes": data.num_classes,
                                                     "classes": data.classes,
                                                     "loss_weights": data.loss_weights,
                                                     "max_time": max_time},
                                   )
            except Exception as err:
                print(err)
#%%

