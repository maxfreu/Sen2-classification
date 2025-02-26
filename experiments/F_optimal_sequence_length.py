import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "optimal_sequence_length"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]


for train_seq_len in (32,64,128):

    version = f"train_seq_len={train_seq_len}"

    for model_config in ("configs/transformer.yaml", "configs/gru.yaml"):
        data, dataconfig = load_data(data_args={"sequence_length": train_seq_len,
                                                "mean": norm_config["mean"],
                                                "stddev": norm_config["stddev"]})

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
                           )
