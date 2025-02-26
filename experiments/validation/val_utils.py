import os
import re
import ast
import yaml
import importlib

from sen2classification.models.transformer import SBERTClassifier
from sen2classification.models.gru import GRU


def get_best_model_checkpoint_path(checkpoint_folder):
    """Returns the full path to the best model checkpoint file.

    Model checkpoint files must be in the form 'epoch=x-step=y.ckpt', in which case the model with highest step
    will be picked - or in the form  'epoch=x-step=y-val_loss=z.ckpt' in which case the model with lowest val_loss is
    chosen.

    Args:
        checkpoint_folder (str): Path to the folder containing checkpoints

    Thanks, GPT!
    """
    # Regular expressions for matching the filenames
    step_pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")
    val_loss_pattern = re.compile(r"epoch=(\d+)-step=(\d+)-val_loss=([\d\.]+)\.ckpt")

    # List of checkpoint files in the folder
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.ckpt')]

    best_checkpoint = None
    best_metric = None

    for filename in checkpoint_files:
        # Try matching with the step pattern
        step_match = step_pattern.match(filename)
        if step_match:
            step = int(step_match.group(2))
            metric = step  # Choosing the highest step as the metric
            if best_checkpoint is None or metric > best_metric:
                best_checkpoint = filename
                best_metric = metric
            continue

        # Try matching with the validation loss pattern
        val_loss_match = val_loss_pattern.match(filename)
        if val_loss_match:
            val_loss = float(val_loss_match.group(3))
            metric = val_loss  # Choosing the lowest val_loss as the metric
            if best_checkpoint is None or metric < best_metric:
                best_checkpoint = filename
                best_metric = metric
            continue

    if best_checkpoint is None:
        raise ValueError("No valid checkpoint files found in the folder.")

    # Return the full path to the best checkpoint
    return os.path.join(checkpoint_folder, best_checkpoint)


def infer_model_name_from_checkpoint_folder(checkpoint_folder):
    parpardir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_folder)))
    return parpardir.split('_')[-1]


def checkpoint_folder_to_configfile(checkpoint_folder):
    return os.path.join(os.path.dirname(os.path.abspath(checkpoint_folder)), "config.yaml")


def instantiate_model_from_checkpoint_folder_and_classname(checkpoint_folder):
    def parse_value(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value

    class_name = infer_model_name_from_checkpoint_folder(checkpoint_folder)
    config_path = checkpoint_folder_to_configfile(checkpoint_folder)
    best_checkpoint_path = get_best_model_checkpoint_path(checkpoint_folder)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if "init_args" in config["model"].keys():
        init_args = {k: parse_value(v) for k, v in config['model']['init_args'].items()}
    else:
        init_args = {k: parse_value(v) for k, v in config['model'].items()}

    if "class_path" in init_args.keys():
        class_path = config['model']['class_path']
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    else:
        if class_name == "SBERTClassifier":
            class_ = SBERTClassifier
        elif class_name == "GRU":
            class_ = GRU
        else:
            raise ValueError("Config file must contain class path or class name must be explicitly given.")

    model = class_.load_from_checkpoint(best_checkpoint_path, **init_args)
    model.eval()
    return model