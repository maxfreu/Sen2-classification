#%%
from sen2classification.models.classifiers import MultiModalTreeClassifier
from sen2classification.datamodules import MultiModalClassificationDataModule
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
# from pytorch_lightning.callbacks import ModelCheckpoint


#%%
class CLIWithWeightedLoss(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.loss_weights", apply_on="instantiate")

    # def fit(self, model, **kwargs):
    #     compiled_model = utils.maybe_compile(model)
    #     self.trainer.fit(compiled_model, **kwargs)


cli = CLIWithWeightedLoss(MultiModalTreeClassifier, MultiModalClassificationDataModule)
