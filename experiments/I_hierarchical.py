import os

import torch.cuda
import yaml
from sen2classification.hierarchical_model import HierarchicalModel
from sen2classification import utils
from experiments.train_and_validate import train, validate, train_and_validate, load_data
from pytorch_lightning.loggers import TensorBoardLogger


logdir = "output"
experiment_name = "hierarchical"
model_config = "/home/max/dr/Sen2-classification/configs/gru.yaml"


with open(f"/home/max/dr/Sen2-classification/configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

#%%
################################################
# 3 classes: background, coniferous, deciduous #
################################################
version = f"3classes"
data_3classes, dataconfig_3classes = load_data(dataconfigfile="configs/3_classes.yaml",
                                               data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

model_3classes, output_folder_3classes, best_model_path_3classes, logger, init_args_3classes = train(
    model_config,
    data_3classes,
    logdir,
    experiment_name,
    version=version,
    model_extra_args={"num_classes": data_3classes.num_classes,
                      "classes": data_3classes.classes,
                      "loss_weights": data_3classes.loss_weights, },
)

#%%
###############
# confiferous #
###############
version = f"coniferous"
data_coniferous, dataconfig_coniferous = load_data(dataconfigfile="configs/coniferous.yaml",
                                                   data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

model_coniferous, output_folder, best_model_path_coniferous, logger, init_args = train(
    model_config,
    data_coniferous,
    logdir,
    experiment_name,
    version=version,
    model_extra_args={"num_classes": data_coniferous.num_classes,
                      "classes": data_coniferous.classes,
                      "loss_weights": data_coniferous.loss_weights, },
)


#%%
#############
# broadleaf #
#############
version = f"broadleaf"
data_broadleaf, dataconfig_broadleaf = load_data(dataconfigfile="configs/broadleaf.yaml",
                                                 data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

model_broadleaf, output_folder, best_model_path_broadleaf, logger, init_args = train(
    model_config,
    data_broadleaf,
    logdir,
    experiment_name,
    version=version,
    model_extra_args={"num_classes": data_broadleaf.num_classes,
                      "classes": data_broadleaf.classes,
                      "loss_weights": data_broadleaf.loss_weights, },
)

#%%
# dataconfigpath_3classes = "/home/max/dr/Sen2-classification/configs/3_classes.yaml"
# checkpoint_3classes = f"/home/max/dr/Sen2-classification/output/hierarchical_GRU/3classes/checkpoints/epoch=12-step=32448.ckpt"
# model_3classes, init_args_3classes = utils.load_model_from_configs_and_checkpoint(
#     model_config=model_config,
#     data_config=dataconfigpath_3classes,
#     checkpoint_path=checkpoint_3classes
# )
#
# dataconfigpath_coniferous = "/home/max/dr/Sen2-classification/configs/coniferous.yaml"
# checkpoint_coniferous = f"/home/max/dr/Sen2-classification/output/hierarchical_GRU/coniferous/checkpoints/epoch=14-step=19890.ckpt"
# model_coniferous, _ = utils.load_model_from_configs_and_checkpoint(
#     model_config=model_config,
#     data_config=dataconfigpath_coniferous,
#     checkpoint_path=checkpoint_coniferous
# )
#
# dataconfigpath_broadleaf = "/home/max/dr/Sen2-classification/configs/broadleaf.yaml"
# checkpoint_broadleaf = f"/home/max/dr/Sen2-classification/output/hierarchical_GRU/broadleaf/checkpoints/epoch=17-step=14184.ckpt"
# model_broadleaf, _ = utils.load_model_from_configs_and_checkpoint(
#     model_config=model_config,
#     data_config=dataconfigpath_broadleaf,
#     checkpoint_path=checkpoint_broadleaf
# )


#%%
######################
# hierarchical model #
######################
version = f"hierarchical"

device = "cuda" if torch.cuda.is_available else "cpu"
model_3classes = model_3classes.to(device)
model_coniferous = model_coniferous.to(device)
model_broadleaf = model_broadleaf.to(device)

model_hierarchical = HierarchicalModel(model_3classes,
                                       [model_coniferous, model_broadleaf],
                                       num_classes=[model_coniferous.num_classes, model_broadleaf.num_classes],
                                       reorder_dict={0:3,
                                                     1:5, 2:6, 3:7, 4:10, 5:12, 6:13,
                                                     7:0, 8:1, 9:2, 10:4, 11:8, 12:9, 13:11},
                                       classes=list(sorted(["BG"] + model_coniferous.classes + model_broadleaf.classes)))

model_hierarchical = model_hierarchical.to(device)

#%%
data_all, dataconfig_all = load_data(dataconfigfile="/home/max/dr/Sen2-classification/configs/14_classes.yaml",
                                     data_args={"mean": norm_config["mean"], "stddev": norm_config["stddev"]})

#%%
model_name = f"{experiment_name}_{utils.classname(model_hierarchical)}"
output_folder = os.path.join(logdir, model_name, version)
logger = TensorBoardLogger(logdir, model_name, version=version)

validate(
    model=model_hierarchical,
    data=data_all,
    output_folder=output_folder,
    logger=logger,
    init_args=init_args_3classes,
    dataconfig=dataconfig_all
)

