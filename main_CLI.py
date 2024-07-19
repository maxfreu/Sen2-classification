import os
import sys

import torch
import datetime
from sen2classification import utils
from sen2classification.datamodules import TimeSeriesClassificationDataModule
from validation_exploratories import validate_exploratories
from validation_treesat import validate_treesat
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


torch.set_float32_matmul_precision("medium")


#%%
class CLIWithWeightedLoss(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """This is here just for plumbing; we link the number of classes in the dataset to 
        the number of classes the model should output and do the same for the loss weights and
        the class names."""
        parser.add_argument("validation")
        parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.init_args.loss_weights", apply_on="instantiate")
        parser.link_arguments("data.classes", "model.init_args.classes", apply_on="instantiate")

    def fit(self, model, **kwargs):
        compiled_model = utils.maybe_compile(model)
        self.trainer.fit(compiled_model, **kwargs)


cli = CLIWithWeightedLoss(datamodule_class=TimeSeriesClassificationDataModule,
                          run=False,
                          save_config_kwargs={"overwrite": False})

cli.trainer.fit(cli.model, cli.datamodule)


if cli.trainer.global_rank > 0:
    sys.exit()

############################
# below is only test stuff #
############################
print("Testing on validation dataloader")

# test with only one device
if cli.trainer.num_devices > 1:
    cli.instantiate_trainer(fast_dev_run=False, devices=1)

# We have linked the data to the model args above. These changes are not reflected in the cli.config
# Therefore we have to redo these changes
cli.config.model.init_args.num_classes = cli.datamodule.num_classes
cli.config.model.init_args.classes = cli.datamodule.classes
cli.config.model.init_args.loss_weights = cli.datamodule.class_weights

# and the best model
best_model_path = cli.trainer.checkpoint_callback.best_model_path
model = type(cli.model).load_from_checkpoint(best_model_path, **cli.config.model.init_args)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

log_dir = cli.trainer.logger.log_dir
version = cli.trainer.logger.version

ret_mode = "last"

for seq_len in (16,32,64):
    cli.datamodule.val_data.return_mode = ret_mode
    cli.datamodule.val_data.sequence_length = seq_len

    val_pred = model.test_on_dataloader(cli.datamodule.val_dataloader(),
                                        log_dir,
                                        version,
                                        seq_len=seq_len,
                                        return_mode=ret_mode)

    assert val_pred is not None

    utils.save_pandas_as_sqlite(os.path.join(log_dir, f"prediction_seq_len={seq_len}_ret_mode={ret_mode}.sqlite"),
                                [val_pred],
                                ["val"],
                                overwrite=True)


###########################################
# independent validation on exploratories #
###########################################
print("Validating on exploratories")

output_folder = os.path.join(log_dir, version)

seq_len = cli.datamodule.sequence_length
class_mapping = cli.datamodule.class_mapping
time_encoding = cli.datamodule.pos_encode
acc_expl, kl_div_expl = validate_exploratories(model,
                                               input_folder=cli.config["validation"]["exploratories_dir"],
                                               class_mapping=class_mapping,
                                               outpath=output_folder,
                                               time_encoding=time_encoding,
                                               seq_len=seq_len)

#####################################
# independent validation on treesat #
#####################################
print("Validating on treesat")
acc_treesat = validate_treesat(model,
                               treesat_dir=cli.config["validation"]["treesat_dir"],
                               class_mapping=class_mapping,
                               outpath=output_folder,
                               version=version,
                               seq_len=seq_len)

report_outfile = os.path.join(log_dir, version, "validation_report.txt")
with open(report_outfile, "w") as f:
    f.write(f"Exploratories accuracy: {acc_expl}\n")
    f.write(f"Exploratories mean KL Divergence: {kl_div_expl}\n")
    f.write(f"Treesat accuracy: {acc_treesat}\n")
