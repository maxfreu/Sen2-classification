import yaml
from experiments.train_and_validate import train_and_validate, load_data

logdir = "output"
experiment_name = "two_years_double_val"


with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

for return_mode in ("single", "double", "random"):
    version = f"return_mode={return_mode}"

    for model_config in ("configs/transformer.yaml", "configs/gru.yaml"):
        data, dataconfig = load_data(data_args={"sequence_length": 128,
                                                "mean": norm_config["mean"],
                                                "stddev": norm_config["stddev"],
                                                "return_mode": return_mode})

        train_and_validate(model_config,
                           data,
                           dataconfig | {"normalization": "223_g-5k"},
                           logdir,
                           experiment_name=experiment_name,
                           version=version,
                           experiment_file=__file__,
                           model_extra_args={"num_classes": data.num_classes,
                                             "classes": data.classes,
                                             "loss_weights": data.loss_weights,},
                           val_return_mode="double"
                           )
