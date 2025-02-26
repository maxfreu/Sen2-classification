import yaml
from experiments.train_and_validate import train_and_validate, load_data


logdir = "output"
experiment_name = "additional_experiments"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

train_seq_len = 32

# time_encoding = "doy"
# for embedding_type in ("bert", "concat"):
#
#     version = f"train-seq-len={train_seq_len}_time-encoding={time_encoding}_embedding-type={embedding_type}"
#
#     for model_config in ("configs/transformer.yaml",):
#         # TODO: adjust model config for gru
#
#         data, dataconfig = load_data(data_args={"sequence_length": train_seq_len,
#                                                 "mean": norm_config["mean"],
#                                                 "stddev": norm_config["stddev"],
#                                                 "time_encoding": time_encoding})
#
#         train_and_validate(model_config,
#                            data,
#                            dataconfig | {"normalization": "223_g-5k"},
#                            logdir,
#                            experiment_name=experiment_name,
#                            version=version,
#                            model_extra_args={"num_classes": data.num_classes,
#                                              "classes": data.classes,
#                                              "loss_weights": data.loss_weights,
#                                              "embedding_type": embedding_type},
#                            trainer_extra_args={"max_steps": max_steps,
#                                                "log_every_n_steps": 400}
#                            )

time_encoding = "absolute"
for embedding_type in ("concat",):

    version = f"train-seq-len={train_seq_len}_time-encoding={time_encoding}_embedding-type={embedding_type}"

    for model_config in ("configs/transformer.yaml",):
        data, dataconfig = load_data(data_args={"sequence_length": train_seq_len,
                                                "mean": norm_config["mean"],
                                                "stddev": norm_config["stddev"],
                                                "time_encoding": time_encoding})

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
                                             "embedding_type": embedding_type},
                           )
