#%%
from sen2classification import utils
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from sen2classification import TreeClassifier
from sen2classification.datamodules import ClassificationDataModule
from omegaconf import OmegaConf
# from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

#%%
parser = ArgumentParser()
parser.add_argument("--config", "-c",type=str, required=True)
args = parser.parse_args()

defaults = OmegaConf.load("default_config.yaml")
settings = OmegaConf.load(args.config)
cfg = OmegaConf.merge(defaults, settings)

logger = TensorBoardLogger(**cfg.tensorboard)

checkpoint_callback = ModelCheckpoint(save_top_k=2, save_last=True, monitor="val_loss")


data = ClassificationDataModule(**cfg.data)
data.setup("fit")

model = TreeClassifier(data=data, **cfg.model)
model = utils.maybe_compile(model)

#%%
trainer = Trainer(**cfg.trainer, logger=logger, callbacks=[checkpoint_callback,])
trainer.fit(model, datamodule=data)

#%%
