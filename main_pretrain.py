import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sen2classification.models.transformer import SBERTPretrain
from sen2classification.datasets import PretrainingDatasetNPZ


torch.set_float32_matmul_precision("medium")

train_split = 0.8
# files = np.array(glob("/data_ssd/bwi/cutouts_2023_npz/*.npz"))
files = np.array(glob("/data_local_ssd/bwi/cutouts_2023_npz/*.npz"))
# files = np.array(glob("/tmp/cutouts_2023_npz/*.npz"))
N = len(files)
np.random.seed(0)
np.random.shuffle(files)
train_files = files[:int(N*train_split)]
val_files   = files[int(N*train_split):]
print(val_files)


batchsize = 960

train_ds = PretrainingDatasetNPZ(train_files,
                           64,
                           batchsize,
                           data_mask_percentage=0.15,
                           time_encoding="absolute")

val_ds = PretrainingDatasetNPZ(train_files,
                           64,
                           batchsize,
                           data_mask_percentage=0.15,
                           time_encoding="absolute")


def collate_fn(batches):
    boa        = torch.from_numpy(np.concatenate([x[0] for x in batches]))
    times      = torch.from_numpy(np.concatenate([x[1] for x in batches]))
    mask       = torch.from_numpy(np.concatenate([x[2] for x in batches]))
    data_mask  = torch.from_numpy(np.concatenate([x[3] for x in batches]))
    return boa, times, mask, data_mask


train_dl = DataLoader(train_ds, batch_size=None, batch_sampler=None, num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)
val_dl   =   DataLoader(val_ds, batch_size=None, batch_sampler=None, num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)

model = SBERTPretrain(lr=3e-4,
                      hidden_dim=96,
                      num_attention_heads=8,
                      transformer_layercount=8,
                      max_embedding_size=3294,
                      cosine_init_period=24000,
                      dropout=0.2)

#%%
logger = TensorBoardLogger(save_dir="output",
                           name="transformer_pretrain")

callbacks = [ModelCheckpoint(dirpath="output/transformer_pretrain",
                             filename="transformer_pretrain-step={step}",
                             monitor="val/loss",
                             save_last=True,
                             save_top_k=2,
                             ),
             LearningRateMonitor()
             ]

trainer = Trainer(precision="bf16-mixed",
                  enable_progress_bar=False,
                  callbacks=callbacks,
                  logger=logger,
                  log_every_n_steps=1000,
                  max_steps=165000+192000,
                  val_check_interval=7000,
                  limit_val_batches=2000)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path="output/transformer_pretrain/last-v2.ckpt")
#%%

