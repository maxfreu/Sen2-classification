# we want to train a hierarchical model for background / leaf / needle, leaf and needle
# consequently we have to load three subsets of the data
# and train three models per architecture
# we want to load the data once and then train on that

# load 3 classes
# train gru, return best model path
# train sbert, return best model path

# load coniferous
# same

# load deciduous
# same

import os
import yaml
import numpy as np
from sen2classification.models.gru import GRU
from sen2classification.models.transformer import SBERTClassifier
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from sen2classification.datamodules import TimeSeriesClassificationDataModule
from sen2classification import utils
from experiments import validate_treesat
from experiments import validate_exploratories
from sen2classification.hierarchical_model import HierarchicalModel


def yaml_to_model(model_class, model_config_yaml):
    with open(model_config_yaml, "r") as f:
        model_args = yaml.safe_load(f)["model"]["init_args"]
    return model_class(**model_args)


def train_model_on_data_subset(logdir, data, data_subset_name, model, trainer_args):
    model_name = utils.classname(model)
    output_folder = os.path.join(logdir, model_name, data_subset_name)
    checkpoint_dir = os.path.join(output_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    utils.copy_file_to_destination(output_folder, __file__)

    logger = TensorBoardLogger(logdir, model_name, version=data_subset_name)

    mc = ModelCheckpoint(dirpath=checkpoint_dir,
                         monitor="val_loss",
                         save_last=True,
                         save_top_k=1,)
    callbacks = [LearningRateMonitor(),
                 # EarlyStopping(patience=5, monitor="val_loss"),
                 mc]
    trainer = Trainer(logger=logger, callbacks=callbacks, **trainer_args)
    trainer.fit(model, data)
    return mc.best_model_path


def train_models(logdir, data_configs, model_dict, trainer_args):
    """Loads data subsets and trains all models specified in the model dict."""
    best_paths = {}
    classes = {}

    for (data_subset_name, data_config_yaml) in data_configs.items():
        with open(data_config_yaml, "r") as f:
            dataconfig = yaml.safe_load(f)["data"]

        data = TimeSeriesClassificationDataModule(**dataconfig)

        best_paths[data_subset_name] = {}
        classes[data_subset_name] = list(sorted(set(data.class_mapping.values())))

        for model_config_yaml in model_dict.values():
            model = utils.instantiate_model_from_config(model_config_yaml)
            model_class_name = utils.classname(model)
            best_model_path = train_model_on_data_subset(logdir, data, data_subset_name, model, trainer_args)
            best_paths[data_subset_name][model_class_name] = best_model_path

    return best_paths, classes


def test_hierarchical_model(hierarchical_model, outpath, class_mapping):
    acc_expl, kl_div_expl = validate_exploratories(hierarchical_model,
                                                   exploratories_dir="/data_hdd/exploratories/S2_L2",
                                                   class_mapping=class_mapping,
                                                   outpath=outpath,
                                                   time_encoding="absolute",
                                                   seq_len=128,
                                                   qai=223,
                                                   mean=np.zeros(10, dtype=np.float32),
                                                   stddev=np.ones(10, dtype=np.float32) * 10000, )

    acc_treesat = validate_treesat(hierarchical_model,
                                   "/data_ssd/treesat/validation_treesat/",
                                   class_mapping,
                                   outpath,
                                   time_encoding="absolute",
                                   seq_len=128,
                                   qai=223,
                                   mean=np.zeros(10, dtype=np.float32),
                                   stddev=np.ones(10, dtype=np.float32) * 10000,
                                   )

    report_outfile = os.path.join(outpath, f"report_{utils.classname(hierarchical_model.parent_model)}.txt")
    with open(report_outfile, "w") as f:
        f.write(f"Exploratories accuracy: {acc_expl}\n")
        f.write(f"Exploratories mean KL Divergence: {kl_div_expl}\n")
        f.write(f"Treesat accuracy: {acc_treesat}\n")


def validate_models(logdir, best_paths, classes, model_default_args, class_mapping):
    class_abbreviations = list(sorted(set(class_mapping.values())))
    for (model_class, kwargs) in model_default_args.items():
        three_classes_ckpt = best_paths["three_classes"][str(model_class)]
        three_classes_model = model_class.load_from_checkpoint(three_classes_ckpt, **kwargs)

        coniferous_ckpt = best_paths["coniferous"][str(model_class)]
        coniferous_model = model_class.load_from_checkpoint(coniferous_ckpt, **kwargs)
        coniferous_classes = len(classes["coniferous"])

        deciduous_ckpt = best_paths["deciduous"][str(model_class)]
        deciduous_model = model_class.load_from_checkpoint(deciduous_ckpt, **kwargs)
        deciduous_classes = len(classes["deciduous"])

        # now some brainfuck, because we have to remap the hierarchical output classes to the class ordering in the
        # original class dict

        hierarchical_model_class_dict = {0: "BG"} | {i + 1: cls for (i, cls) in enumerate(classes["coniferous"] + classes["deciduous"])}
        class_remapping = {i:class_abbreviations.index(abbrev) for (i, abbrev) in hierarchical_model_class_dict.items()}

        model = HierarchicalModel(three_classes_model,
                                  (coniferous_model, deciduous_model),
                                  [coniferous_classes, deciduous_classes],
                                  reorder_dict=class_remapping)

        outpath = os.path.join(logdir, f"{str(model_class)}_hierarchical")

        test_hierarchical_model(model, outpath, class_mapping)


def main():
    """Trains a hierarchical GRU and SBERT for comparison with direct classification."""
    logdir = "output"
    max_steps = 75000 - 1
    trainer_args = {"enable_progress_bar": False,
                    "max_steps": max_steps,
                    "precision": "16-mixed",
                    "log_every_n_steps": 400}

    data_configs = {"three_classes": "configs/3_classes.yaml",
                    "coniferous": "configs/coniferous.yaml",
                    "deciduous": "configs/deciduous.yaml"}

    model_default_args = {GRU: "configs/gru.yaml", SBERTClassifier: "configs/transformer.yaml"}

    best_paths, classes = train_models("output", data_configs, model_default_args, trainer_args)

    # laod original class mapping
    with open("../configs/14_classes.yaml", "r") as f:
        class_mapping = yaml.safe_load(f)["data"]["class_mapping"]

    validate_models(logdir, best_paths, classes, model_default_args, class_mapping)


main()
