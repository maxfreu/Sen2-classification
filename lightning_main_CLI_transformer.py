import os
import sys
import torch
from sen2classification import utils
from sen2classification.classifiers import SBERTClassifier, MockSBERTClassifier
from sen2classification.datamodules import TimeSeriesClassificationDataModule, MockTimeSeriesClassificationDataModule
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import sqlite3


#%%
def save_pandas_as_sqlite(outfile, train_df, val_df, overwrite=False):
    if os.path.exists(outfile) and not overwrite:
        raise RuntimeError(f"Output file {outfile} exists. Set overwrite=True to if needed.")
    else:
        if os.path.exists(outfile) and overwrite:
            os.remove(outfile)
        conn = sqlite3.connect(outfile)
        train_df.to_sql(name="train", con=conn)
        val_df.to_sql(name="val", con=conn)
        conn.close()


class CLIWithWeightedLoss(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """This is here just for plumbing; we link the number of classes in the dataset to 
        the number of classes the model should output and do the same for the weights and
        the class names."""
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.loss_weights", apply_on="instantiate")
        parser.link_arguments("data.classes", "model.classes", apply_on="instantiate")

    def fit(self, model, **kwargs):
        compiled_model = utils.maybe_compile(model)
        self.trainer.fit(compiled_model, **kwargs)


cli = CLIWithWeightedLoss(SBERTClassifier,
                          TimeSeriesClassificationDataModule,
                          run=False,
                          save_config_kwargs={"overwrite": False})

# using torchmetrics currently breaks compilation
# compiled_model = utils.maybe_compile(cli.model)
cli.trainer.fit(cli.model, cli.datamodule)

############################
# below is only test stuff #
############################

# test with only one device
if cli.trainer.num_devices > 1:
    cli.instantiate_trainer(fast_dev_run=False, devices=1)

# test only on the last year
cli.datamodule.val_data.return_mode = "last"

# and the best model
best_model_path = cli.trainer.checkpoint_callback.best_model_path
model = cli.model_class.load_from_checkpoint(best_model_path, **cli.config.model, classes=cli.datamodule.classes)
cli.trainer.test(model, cli.datamodule)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
train_pred = model.predict_dataset(cli.datamodule.training_data)  # pandas dataframe
val_pred   = model.predict_dataset(cli.datamodule.val_data)

out_dir = cli.trainer.logger.log_dir
save_pandas_as_sqlite(os.path.join(out_dir, "prediction.sqlite"), train_pred, val_pred, overwrite=True)
