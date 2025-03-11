import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "normalization"


for normalization in ("10k", "223", "223_same", "223_g-5k", "223_g-5k_same", "all", "all_same"):
    with open(f"configs/statistics_{normalization}.yaml") as f:
        norm_config = yaml.safe_load(f)["data"]

    version = f"data_normalization={normalization}"

    for model_config in ("configs/gru.yaml", "configs/transformer.yaml"):
        data, dataconfig = load_data(overwrite_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

        train_and_validate(model_config,
                           data,
                           dataconfig | {"normalization": normalization},
                           logdir,
                           experiment_name=experiment_name,
                           version=version,
                           experiment_file=__file__,
                           model_extra_args={"num_classes": data.num_classes,
                                             "classes": data.classes,
                                             "loss_weights": data.loss_weights, },
                           )
