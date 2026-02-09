import os
import yaml
import itertools
from experiments.train_and_validate import train_and_validate, load_data


myid = int(os.environ["SLURM_ARRAY_TASK_ID"])

logdir = "output"
experiment_name = "time_encoding"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

model_config = ("configs/gru_best.yaml", "configs/transformer_best.yaml")
time_encoding  = ("doy", "absolute")
params = list(itertools.product(model_config, time_encoding))
model_config, time_encoding = params[myid]

if time_encoding == "doy":
    max_time = 366 + 1
else:
    max_time = 8*366 + 1

return_mode = "single"

version = f"time_encoding={time_encoding}"

data, dataconfig = load_data(overwrite_args={"time_encoding": time_encoding,
                                             "return_mode": return_mode,
                                             "mean": norm_config["mean"],
                                             "stddev": norm_config["stddev"],
                                             "where": "(time < 1609459200)",
                                             "val_where": "(1609459200 <= time and time < 1672531200 and present_2022=1)"
                                             })

train_and_validate(model_config,
                   data,
                   dataconfig | {"normalization": "223_g-5k"},
                   logdir,
                   experiment_name=experiment_name,
                   version=version,
                   experiment_file=__file__,
                   model_extra_args={"num_classes": data.num_classes,
                                     "classes": data.classes,
                                     "max_time": max_time},
                   val_return_mode=return_mode,
                   val_years=(2021, 2022)
                   )
