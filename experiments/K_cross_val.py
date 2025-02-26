import os
import yaml
import torch
import numpy as np
import duckdb
import gc

from copy import deepcopy

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from sen2classification.utils import k_fold_generator_list
from sen2classification.datasets import InMemoryTimeSeriesDataset
from sen2classification.datamodules import seed_worker

from experiments.train_and_validate import train_and_validate, load_data


def get_tree_ids(file):
    ids = duckdb.query(f"select distinct(tree_id) from '{file}' order by tree_id asc").fetchnumpy()["tree_id"]
    return [int(id) for id in ids]  # ids are np.int32 which breaks yaml serialization

#%%
# class CrossValDataModule(LightningDataModule):
#     def __init__(self, dataset, train_indices, val_indices, mean, stddev):
#         super().__init__()
#         self.dataset = dataset
#         self.val_data = dataset
#         self.train_data = dataset
#         self.train_indices = train_indices
#         self.val_indices = val_indices
#         self.mean = np.array(mean)
#         self.stddev = np.array(stddev)
#         self.inv_stddev = 1 / (self.stddev + 1e-9)
#         self.return_mode = self.dataset.return_mode
#         self.return_year = self.dataset.return_year
#         self.time_encoding = self.dataset.time_encoding
#         self.classes = self.dataset.classes
#         self.class_mapping = self.dataset.class_mapping
#         self.quality_mask = 31  # TODO: remove hard coding
#
#     def train_augmentation(self, boa_observation):
#         normalized_obs = (boa_observation - self.mean) * self.inv_stddev
#         return normalized_obs * (0.98 + np.random.rand(10) * 0.04)
#
#     def val_augmentation(self, boa_observation):
#         normalized_obs = (boa_observation - self.mean) * self.inv_stddev
#         return normalized_obs
#
#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         self.dataset.tree_ids = self.train_indices
#         self.dataset.augmentation = self.train_augmentation
#
#         g = torch.Generator()
#         g.manual_seed(0)
#
#         return torch.utils.data.DataLoader(
#             self.dataset,
#             pin_memory=True,
#             shuffle=True,
#             batch_size=dataconfig["batch_size"],
#             num_workers=dataconfig["num_workers"],
#             persistent_workers=False,
#             generator=g,
#             worker_init_fn=seed_worker
#         )
#
#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         self.dataset.tree_ids = self.val_indices
#         self.dataset.augmentation = self.val_augmentation
#
#         g = torch.Generator()
#         g.manual_seed(0)
#
#         return DataLoader(
#             self.dataset,
#             pin_memory=True,
#             shuffle=False,
#             batch_size=dataconfig["batch_size"],
#             num_workers=dataconfig["num_workers"],
#             persistent_workers=False,
#             generator=g,
#             worker_init_fn=seed_worker
#         )
#
#
with open("/home/max/dr/Sen2-classification/configs/14_classes.yaml") as f:
    dataconfig = yaml.safe_load(f)["data"]

with open(f"/home/max/dr/Sen2-classification/configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]
#
# dataset = InMemoryTimeSeriesDataset(
#     input_filepath=dataconfig["input_file"],
#     dbname=dataconfig["dbname"],
#     augmentation=np.identity,
#     sequence_length=dataconfig["sequence_length"],
#     quality_mask=dataconfig["quality_mask"],
#     class_mapping=dataconfig["class_mapping"],
#     return_mode=dataconfig["return_mode"],
#     return_year=None,
#     time_encoding=dataconfig["time_encoding"],
#     # where="tree_id > -10 and tree_id < 10000"
# )

logdir = "output"
experiment_name = "cross_validation"

model_config = "configs/gru.yaml"

tree_ids = get_tree_ids(dataconfig["input_file"])
classes = list(sorted(set(dataconfig["class_mapping"].values())))
num_classes = len(classes)
num_folds = 5
gen = k_fold_generator_list(tree_ids, num_folds, test_fraction=0.2)

#%%
# for i in range(num_folds):
#     data = None
#     gc.collect()
#
#     train_indices, val_indices = next(gen)

i = 0
for j in range(int(os.environ["SLURM_ARRAY_TASK_ID"])):
    train_indices, val_indices = next(gen)
    i = j

data, dataconfig = load_data(data_args={
    "mean": norm_config["mean"],
    "stddev": norm_config["stddev"],
    "train_ids": list(train_indices),
    "val_ids": list(val_indices)
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
    model_extra_args={
        "num_classes": data.num_classes,
        "classes": data.classes,
        "loss_weights": data.loss_weights,
        "focalloss_gamma": 0},
    val_return_mode="single",
)
