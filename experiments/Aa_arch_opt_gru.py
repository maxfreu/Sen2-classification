import os
import yaml
import itertools
from experiments.train_and_validate import train_and_validate, load_data

myid = int(os.environ["SLURM_ARRAY_TASK_ID"])

logdir = "output"
experiment_name = "architecture_opt"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

data, dataconfig = load_data(overwrite_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

num_layers = [2, 4]
hidden_dim = [128, 256, 512]
fc_size = [128, 256, 512]
bidirectional = [True, False]

params = list(itertools.product(num_layers, hidden_dim, fc_size, bidirectional))
num_layers, hidden_dim, fc_size, bidirectional = params[myid]

version = f"num_layers={num_layers}-hidden_dim={hidden_dim}-fc_size={fc_size}-bi={bidirectional}"

train_and_validate("configs/gru.yaml",
                   data,
                   dataconfig | {"normalization": "223_g-5k"},
                   logdir,
                   experiment_name=experiment_name,
                   version=version,
                   experiment_file=__file__,
                   model_extra_args={"num_classes": data.num_classes,
                                     "classes": data.classes,
                                     "num_layers": num_layers,
                                     "fc_size": fc_size,
                                     "hidden_dim": hidden_dim,
                                     "bidirectional": bidirectional
                                     },
                   )
