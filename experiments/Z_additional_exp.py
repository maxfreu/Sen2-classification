import os
import yaml
from experiments.train_and_validate import train_and_validate, load_data

logdir = "output"
experiment_name = "additional_experiments"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

# myid = int(os.environ["SLURM_ARRAY_TASK_ID"])

model_config = "configs/transformer_best.yaml"
embedding_type = "bert"
augmentation_strength = 0.02
time_shift = 5
focalloss_gamma = 1
return_mode = "single"

version = f"small_augmentation-strength={augmentation_strength}_embedding-type={embedding_type}_time-shift={time_shift}_fl-gamma={focalloss_gamma}_return-mode={return_mode}"

data, dataconfig = load_data(overwrite_args={"time_encoding": "doy",
                                             "time_shift": time_shift,
                                             "mean": norm_config["mean"],
                                             "stddev": norm_config["stddev"],
                                             "return_mode": return_mode,
                                             "augmentation_strength": augmentation_strength,
                                             "where": "(time < 1609459200)",
                                             "val_where": "(1609459200 <= time and time < 1672531200 and present_2022=1)"})

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
                                     "embedding_type": embedding_type,
                                     "max_time": 367,
                                     "focalloss_gamma": focalloss_gamma
                                     },
                   val_years=(2021, 2022)
                   )
