import os
import yaml
import duckdb
from sen2classification.utils import k_fold_generator_list
from experiments.train_and_validate import train_and_validate, load_data


def get_tnrs(file):
    ids = duckdb.query(f"select distinct(tnr) from '{file}' order by tnr asc").fetchnumpy()["tnr"]
    return [int(id) for id in ids]  # ids are np.int32 which breaks yaml serialization


with open("/home/max/dr/Sen2-classification/configs/14_classes.yaml") as f:
    dataconfig = yaml.safe_load(f)["data"]

with open(f"/home/max/dr/Sen2-classification/configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

logdir = "output"
experiment_name = "final_cross_validation"
model_config = "configs/gru_best.yaml"
tract_ids = get_tnrs(dataconfig["input_file"])
num_folds = 5
gen = k_fold_generator_list(tract_ids, num_folds, test_fraction=0.2)

i = 0
for j in range(int(os.environ["SLURM_ARRAY_TASK_ID"])):
    train_indices, val_indices = next(gen)
    i = j

data, dataconfig = load_data(overwrite_args={
    "mean": norm_config["mean"],
    "stddev": norm_config["stddev"],
    "train_ids": list(train_indices),
    "val_ids": list(val_indices),
})

version = f"cross_val={i}"

train_and_validate(
    model_config,
    data,
    dataconfig | {"normalization": "223_g-5k"},
    logdir,
    experiment_name=experiment_name,
    version=version,
    experiment_file=__file__,
    trainer_extra_args={"max_steps": 120000},
    model_extra_args={
        "num_classes": data.num_classes,
        "classes": data.classes,
        },
    val_return_mode="single",
)
