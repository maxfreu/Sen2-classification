import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "layernorm_on_input"


with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]


for use_layernorm in (True, False):
    version = f"layernorm={use_layernorm}"

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
                                             "layernorm_on_input": use_layernorm},
                           )
