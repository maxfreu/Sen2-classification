import os
import sys
import torch
from sen2classification import utils
from sen2classification.classifiers import SBERTClassifier, MockSBERTClassifier
from sen2classification.datamodules import TimeSeriesClassificationDataModule, MockTimeSeriesClassificationDataModule
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

torch.set_float32_matmul_precision("medium")

#%%
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

# and the best model
best_model_path = cli.trainer.checkpoint_callback.best_model_path
model = cli.model_class.load_from_checkpoint(best_model_path, **cli.config.model, classes=cli.datamodule.classes)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

out_dir = cli.trainer.logger.log_dir

for ret_mode in ("random", "last"):
    for seq_len in (16, 32, 64, 128):
        cli.datamodule.training_data.return_mode = ret_mode
        cli.datamodule.val_data.return_mode = ret_mode

        cli.datamodule.training_data.sequence_length = seq_len
        cli.datamodule.val_data.sequence_length = seq_len

        train_pred = model.test_on_dataloader(cli.datamodule.train_dataloader(), cli.trainer.logger, seq_len=seq_len, return_mode=ret_mode)  # pandas dataframe
        val_pred   = model.test_on_dataloader(cli.datamodule.val_dataloader(),   cli.trainer.logger, seq_len=seq_len, return_mode=ret_mode)

        utils.save_pandas_as_sqlite(os.path.join(out_dir, f"prediction_seq_len={seq_len}_ret_mode={ret_mode}.sqlite"), train_pred, val_pred, overwrite=True)
