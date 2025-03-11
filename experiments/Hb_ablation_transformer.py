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

params = list(itertools.product((4,6,8), (256,), (4,8,16)))
num_layers, hidden_dim, num_attention_heads = params[myid]

# for num_layers in (2, 4, 6, 8):
#     for hidden_dim in (32, 128, 512, 1024):
#         for num_attention_heads in (4, 8, 16):
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
                                     "loss_weights": data.loss_weights,
                                     "num_layers": num_layers,
                                     "hidden_dim": hidden_dim,
                                     "num_attention_heads": num_attention_heads
                                     },
                   )
