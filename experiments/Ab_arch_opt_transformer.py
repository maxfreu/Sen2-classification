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

num_layers = (4,8)
hidden_dim = (256, 512)
num_attention_heads = (4, 8, 16, 32)

params = list(itertools.product(num_layers, hidden_dim, num_attention_heads))
num_layers, hidden_dim, num_attention_heads = params[myid]

version = f"num_layers={num_layers}-hidden_dim={hidden_dim}-attention_heads={num_attention_heads}"

train_and_validate("configs/transformer.yaml",
                   data,
                   dataconfig | {"normalization": "223_g-5k"},
                   logdir,
                   experiment_name=experiment_name,
                   version=version,
                   experiment_file=__file__,
                   model_extra_args={"num_classes": data.num_classes,
                                     "classes": data.classes,
                                     "num_layers": num_layers,
                                     "hidden_dim": hidden_dim,
                                     "num_attention_heads": num_attention_heads
                                     },
                   )
