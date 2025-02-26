import yaml
from experiments.train_and_validate import train_and_validate, load_data

logdir = "output"
experiment_name = "data_subset"


subsets = {"all": "",
           "continuous": "present_2022 = 1",
           "pure": "is_pure = 1",
           "large": "crown_area_m2 > 10 or crown_area_m2 = 0",
           "continuous_pure": "present_2022 = 1 and is_pure = 1",
           "continuous_pure_large": "present_2022 = 1 and is_pure = 1 and (crown_area_m2 > 10 or crown_area_m2 = 0)",
           }

with open("/home/max/dr/Sen2-classification/configs/statistics_223_g-5k.yaml", "r") as f:
    norm = yaml.safe_load(f)["data"]
    mean = norm["mean"]
    stddev = norm["stddev"]

for name, where in subsets.items():
    # train on data subset, but validate against all data
    data, dataconfig = load_data(data_args={"where": where, "val_where": "", "mean": mean, "stddev": stddev, "sequence_length": 32})

    version = f"subset={name}"

    for model_config in ("configs/gru.yaml", "configs/transformer.yaml"):
        train_and_validate(model_config,
                           data,
                           dataconfig | {"normalization": "223_g-5k", "subset": name},
                           logdir,
                           experiment_name=experiment_name,
                           version=version,
                           experiment_file=__file__,
                           model_extra_args={"num_classes": data.num_classes,
                                             "classes": data.classes,
                                             "loss_weights": data.loss_weights, },
                           )
