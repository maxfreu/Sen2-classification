import os
import yaml
import torch
import random
import numpy as np
from glob import glob

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from sen2classification.models.transformer import SBERTPretrain, SBERTClassifier, SBERTClassification
from sen2classification.datasets import PretrainingDatasetNPZ
from sen2classification import utils
from sen2classification.plotting import plot_confusion_matrices

from experiments.train_and_validate import load_data, write_report, save_classes
from experiments.validation.validation_treesat import validate_treesat
from experiments.validation.validation_exploratories import validate_exploratories


def collate_fn(batches):
    boa        = torch.from_numpy(np.concatenate([x[0] for x in batches]))
    times      = torch.from_numpy(np.concatenate([x[1] for x in batches]))
    mask       = torch.from_numpy(np.concatenate([x[2] for x in batches]))
    data_mask  = torch.from_numpy(np.concatenate([x[3] for x in batches]))
    return boa, times, mask, data_mask


#%%
################
# Common setup #
################
logdir = "output/pretrain"
model_name = "SBERTPretrain"
version = "standard_model"

os.makedirs(logdir, exist_ok=True)
os.makedirs(logdir+f"/{version}", exist_ok=True)

###############
# Pretraining #
###############
torch.set_float32_matmul_precision("medium")
random.seed(1337)
np.random.seed(1337)

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

train_split = 0.8
# files = np.array(glob("/data_ssd/bwi/cutouts_2023_npz/*.npz"))
files = np.array(glob("/data_local_ssd/bwi/cutouts_2023_npz/*.npz"))
# files = np.array(glob("/tmp/cutouts_2023_npz/*.npz"))
N = len(files)
np.random.shuffle(files)
train_files = files[:int(N*train_split)]
val_files   = files[int(N*train_split):]
print(val_files)


batchsize = 960

train_ds = PretrainingDatasetNPZ(train_files,
                                 64,
                                 batchsize,
                                 data_mask_percentage=0.2,
                                 time_encoding="absolute")

val_ds = PretrainingDatasetNPZ(val_files,
                               64,
                               batchsize,
                               data_mask_percentage=0.2,
                               time_encoding="absolute")

train_dl = DataLoader(train_ds, batch_size=None, batch_sampler=None, num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)
val_dl   =   DataLoader(val_ds, batch_size=None, batch_sampler=None, num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)

with open("/home/max/dr/Sen2-classification/configs/transformer.yaml") as f:
    model_config = yaml.safe_load(f)

init_args = model_config["model"]["init_args"]
init_args = init_args | {"cosine_init_period": 24000}
init_args["lr"] = float(init_args["lr"])
init_args["max_time"] = 10*366

del init_args["use_weighted_loss"]

model = SBERTPretrain(**init_args)
# model = torch.compile(model)

#%%
logger = TensorBoardLogger(logdir, version, version=model_name)

output_folder = os.path.join(logdir, version, model_name)
checkpoint_dir = os.path.join(output_folder, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

mc = ModelCheckpoint(dirpath=checkpoint_dir,
                     monitor="val_loss",
                     save_last=True,
                     save_top_k=2,
                     )

callbacks = [mc,
             LearningRateMonitor(),
             EarlyStopping(patience=20, monitor="val_loss"),
             ]

trainer = Trainer(precision="16-mixed",
                  enable_progress_bar=False,
                  callbacks=callbacks,
                  logger=logger,
                  log_every_n_steps=1000,
                  max_steps=165000+192000,
                  val_check_interval=7000,
                  fast_dev_run=False,
                  limit_val_batches=2000)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


#%%
#############################
# Downstream Model Training #
#############################
with open(f"/home/max/dr/Sen2-classification/configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

data, dataconfig = load_data(dataconfigfile="/home/max/dr/Sen2-classification/configs/14_classes.yaml",
                             data_args={"mean": norm_config["mean"],
                                        "stddev": norm_config["stddev"]})

init_args["cosine_init_period"] = 0
init_args["pretrained_model_path"] = mc.best_model_path
# init_args["pretrained_model_path"] = "/home/max/dr/Sen2-classification/output/transformer_pretrain/transformer_pretrain-step=step=348652.ckpt"
init_args["num_classes"] = data.num_classes
init_args["classes"] = data.classes
classifier = SBERTClassifier(**init_args)
model_name = utils.classname(classifier)

# classifier = torch.compile(classifier)

output_folder = os.path.join(logdir, version, model_name)
checkpoint_dir = os.path.join(output_folder, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

utils.copy_file_to_destination(output_folder, __file__)
#
logger = TensorBoardLogger(logdir, version, version=model_name)
#
mc = ModelCheckpoint(dirpath=checkpoint_dir,
                     monitor="val_loss",
                     save_last=True,
                     save_top_k=2,)
callbacks = [LearningRateMonitor(),
             EarlyStopping(patience=10, monitor="val_loss"),
             mc]

trainer = Trainer(logger=logger,
                  precision="16-mixed",
                  callbacks=callbacks,
                  enable_progress_bar=False,
                  max_steps=75001,
                  fast_dev_run=False,
                  log_every_n_steps=400,
                  )

trainer.fit(classifier, data)

###########
# TESTING #
###########
best_model_path = mc.best_model_path
# best_model_path = "/home/max/dr/Sen2-classification/output/SBERT_pretained_SBERTClassifier/1/checkpoints/epoch=17-step=44928.ckpt"
metrics = {}

classifier = classifier.eval()

state_dict = torch.load(best_model_path, map_location=classifier.device)["state_dict"]
classifier.load_state_dict(state_dict)
classifier = classifier.to("cuda" if torch.cuda.is_available() else "cpu")
classifier = classifier.eval()

for seq_len in (16, 32, 64):
    ##############
    # on dataset #
    ##############
    years = (2018, 2020, 2022)
    accs_ds = []  # list of accuracies for each year
    for year in years:
        data.val_data.sequence_length = seq_len
        data.val_data.return_year = year
        data.val_data.return_mode = "single"

        acc_ds, cm, val_pred = classifier.test_on_dataloader(data.val_dataloader())
        accs_ds.append(acc_ds)

        plot_confusion_matrices(os.path.join(os.path.join(output_folder, "plots")), cm, data.class_mapping, "S2GNFI", f"seq_len={seq_len}_year={year}")

        assert val_pred is not None

        utils.save_pandas_as_sqlite(os.path.join(output_folder, f"prediction_seq_len={seq_len}_year={year}.sqlite"),
                                    [val_pred],
                                    ["val"],
                                    overwrite=True)

    ###########################################
    # independent validation on exploratories #
    ###########################################
    print("Validating on exploratories")
    acc_expl, kl_div_expl = validate_exploratories(classifier,
                                                   exploratories_dir="/data_hdd/exploratories/S2_L2",
                                                   class_mapping=data.class_mapping,
                                                   outpath=os.path.join(output_folder, "plots"),
                                                   time_encoding=data.time_encoding,
                                                   seq_len=seq_len,
                                                   qai=data.quality_mask,
                                                   mean=data.mean,
                                                   stddev=data.stddev)

    #####################################
    # independent validation on treesat #
    #####################################
    print("Validating on treesat")
    acc_treesat, cm = validate_treesat(classifier,
                                       treesat_dir="/data_ssd/treesat/validation_treesat",
                                       class_mapping=data.class_mapping,
                                       time_encoding=data.time_encoding,
                                       seq_len=seq_len,
                                       qai=data.quality_mask,
                                       mean=data.mean,
                                       stddev=data.stddev
                                       )

    plot_confusion_matrices(os.path.join(output_folder, "plots"),
                            cm,
                            data.class_mapping, "treesat", f"seq_len={seq_len}")

    write_report(output_folder, f"seq_len={seq_len}", years, accs_ds, acc_expl, kl_div_expl, acc_treesat)

    metrics = metrics | {#f"years=": years,
                         f"acc_ds={seq_len}": np.mean(accs_ds),
                         f"acc_expl_seq_len={seq_len}": acc_expl,
                         f"acc_treesat_seq_len={seq_len}": acc_treesat,
                         f"kl_div_expl_seq_len={seq_len}": kl_div_expl}

save_classes(output_folder, data.classes)

hyper_params = init_args | {"model": utils.classname(classifier)} | dataconfig
print("hyper params:", hyper_params)
# log the hyper params twice, because once is not enough...
logger.log_hyperparams(hyper_params, metrics=metrics)
logger.log_hyperparams(hyper_params, metrics=metrics)

utils.save_dict_to_yaml({"model": init_args, "data": dataconfig}, os.path.join(output_folder, "config.yaml"))
