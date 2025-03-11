import yaml
from experiments.train_and_validate import train_and_validate, load_data


experiment_name = "weighted_loss"
logdir = "output"


with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]


# for use_weighted_loss in (True, False):
for use_weighted_loss in (True,):
    version = f"weighted_loss={use_weighted_loss}"

    for model_config in ("configs/gru.yaml", "configs/transformer.yaml"):
        data, dataconfig = load_data(overwrite_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

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
                                             "use_weighted_loss": use_weighted_loss},
                           )
