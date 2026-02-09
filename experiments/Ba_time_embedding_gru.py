import os
import yaml
import itertools
from experiments.train_and_validate import train_and_validate, load_data

myid = int(os.environ["SLURM_ARRAY_TASK_ID"])

logdir = "output"
experiment_name = "time_embedding"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

data, dataconfig = load_data(overwrite_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

embedding_type = ("bert", "concat")
embedding_dim  = (64, 128, 256)

params = list(itertools.product(embedding_type, embedding_dim))
embedding_type, embedding_dim = params[myid]

version = f"embedding_type={embedding_type}_embedding_dim={embedding_dim}_seq_len={dataconfig['sequence_length']}"

# skip combinations that make no sense
if (embedding_type is None and embedding_dim is not None) or (embedding_type is not None and embedding_dim is None):
    exit(0)

train_and_validate("configs/gru_best.yaml",
                   data,
                   dataconfig | {"normalization": "223_g-5k"},
                   logdir,
                   experiment_name=experiment_name,
                   version=version,
                   experiment_file=__file__,
                   model_extra_args={"num_classes": data.num_classes,
                                     "classes": data.classes,
                                     "embedding_type": embedding_type,
                                     "embedding_dim": embedding_dim},
                   )
