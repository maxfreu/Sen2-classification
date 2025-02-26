import os
import yaml
import itertools
from experiments.train_and_validate import train_and_validate, load_data

myid = int(os.environ["SLURM_ARRAY_TASK_ID"])

logdir = "output"
experiment_name = "architecture_opt"

with open(f"configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

data, dataconfig = load_data(data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

# params = list(itertools.product((2,4,6,8), (64,128,512), (128,512,1024)))
params = [(4, 1024, 512)]
num_layers, hidden_dim, fc_size = params[myid]

# for num_layers in (2,4,6,8):
#     for hidden_dim in (64, 128, 512):
#         for fc_size in (128, 512, 1024):
version = f"num_layers={num_layers}-hidden_dim={hidden_dim}-fc_size={fc_size}"

train_and_validate("configs/gru.yaml",
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
                                     "fc_size": fc_size,
                                     "hidden_dim": hidden_dim
                                     },
                   )

# for num_layers in (2,3,4,6,8):
#     version = f"num_layers={num_layers}"
#
#     train_and_validate("configs/gru.yaml",
#                        data,
#                        dataconfig | {"normalization": "223_g-5k"},
#                        logdir,
#                        experiment_name=experiment_name,
#                        version=version,
#                        experiment_file=__file__,
#                        model_extra_args={"num_classes": data.num_classes,
#                                          "classes": data.classes,
#                                          "loss_weights": data.loss_weights,
#                                          "num_layers": num_layers
#                                          },
#                        )
#
#
# for hidden_dim in (64, 128, 512):
#     version = f"hidden_dim={hidden_dim}"
#
#     train_and_validate("configs/gru.yaml",
#                        data,
#                        dataconfig | {"normalization": "223_g-5k"},
#                        logdir,
#                        experiment_name=experiment_name,
#                        version=version,
#                        experiment_file=__file__,
#                        model_extra_args={"num_classes": data.num_classes,
#                                          "classes": data.classes,
#                                          "loss_weights": data.loss_weights,
#                                          "hidden_dim": hidden_dim
#                                          },
#                        )
#
#
# for fc_size in (128, 512, 1024):
#     version = f"fc_size={fc_size}"
#
#     train_and_validate("configs/gru.yaml",
#                        data,
#                        dataconfig | {"normalization": "223_g-5k"},
#                        logdir,
#                        experiment_name=experiment_name,
#                        version=version,
#                        experiment_file=__file__,
#                        model_extra_args={"num_classes": data.num_classes,
#                                          "classes": data.classes,
#                                          "loss_weights": data.loss_weights,
#                                          "fc_size": fc_size
#                                          },
#                        )
