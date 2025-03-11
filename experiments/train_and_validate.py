import os
import time
from pathlib import Path

import yaml
import torch
import random
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from sen2classification import utils
from sen2classification.schedulerfreemodelcheckpoint import SchedulerFreeModelCheckpoint
from sen2classification.datamodules import TimeSeriesClassificationDataModule, seed_worker
from .validation.validation_treesat import validate_treesat
from .validation.validation_exploratories import validate_exploratories
from .validation.val_utils import checkpoint_folder_to_configfile, instantiate_model_from_checkpoint_folder_and_classname


def load_data(dataconfigfile = "/home/max/dr/Sen2-classification/configs/14_classes.yaml", overwrite_args=None):
    if overwrite_args is None:
        overwrite_args = {}

    with open(dataconfigfile, "r") as f:
        dataconfig = yaml.safe_load(f)["data"]

    dataconfig = dataconfig | overwrite_args

    if not os.path.exists(dataconfig["input_file"]):
        print("[WARNING] Falling back to data loading from remote HDD.")
        dataconfig["input_file"] = "/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet"

    data = TimeSeriesClassificationDataModule(**dataconfig)
    data.setup()
    return data, dataconfig


def save_classes(outpath, classes):
    with open(os.path.join(outpath, "classes.txt"), "w") as f:
        for (i, c) in zip(range(len(classes)), classes):
            f.write(f"{i},{c}\n")


def write_report(output_folder, qualifier, years, metrics):
    report_outfile = os.path.join(output_folder, f"validation_report_{qualifier}.txt")
    with open(report_outfile, "w") as f:
        f.write(f"Years: {years}\n")
        for (k,v) in metrics.items():
            f.write(f"{k}: {v}\n")


def set_random_seeds(seed=1337):
    """Set the random seeds for torch, numpy, and random for reproducibility."""
    torch.set_float32_matmul_precision("medium")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model_config, data, logdir, experiment_name, version, model_extra_args={}, trainer_extra_args={},
          experiment_file=""):
    """Train a model with the given configuration and log details."""
    set_random_seeds()

    model, init_args = utils.instantiate_model_from_config(model_config, **model_extra_args)
    model_name = f"{experiment_name}_{utils.classname(model)}"

    # try:
    #     model = torch.compile(model)
    # except Exception as err:
    #     print("[WARNING] Could not compile model due to following error:")
    #     print(err)

    output_folder = os.path.join(logdir, model_name, version)
    checkpoint_dir = os.path.join(output_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    utils.copy_file_to_destination(output_folder, __file__)
    if experiment_file:
        utils.copy_file_to_destination(output_folder, experiment_file)

    logger = TensorBoardLogger(logdir, model_name, version=version)
    mc = SchedulerFreeModelCheckpoint(dirpath=checkpoint_dir, filename="{epoch}-{step}-{val_loss:.4f}", monitor="val_loss", save_last=True, save_top_k=2)
    callbacks = [LearningRateMonitor(), EarlyStopping(patience=10, monitor="val_loss"), mc]

    trainer_args = {
        "max_steps": 100000,
        "log_every_n_steps": 400,
        "precision": "16-mixed"}

    trainer_args = trainer_args | trainer_extra_args

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        **trainer_args)
    trainer.fit(model, data)

    # restore best weights
    state_dict = torch.load(mc.best_model_path, map_location=model.device)["state_dict"]
    model.load_state_dict(state_dict)

    return model, output_folder, mc.best_model_path, logger, init_args


def validate(checkpoint_folder, val_ds, return_mode="single", val_years=(2018, 2020, 2022), num_workers=0):
    """Validate the best trained model and compute metrics."""
    print("Starting validation")
    tstart = time.time()

    # load config stuff
    config_path = checkpoint_folder_to_configfile(checkpoint_folder)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if "init_args" in config["model"].keys():
        model_init_args = config["model"]["init_args"]
    else:
        model_init_args = config["model"]
    dataconfig = config["data"]
    dataconfig["mean"] = np.array(dataconfig["mean"])
    dataconfig["stddev"] = np.array(dataconfig["stddev"])

    # load best model from checkpoint
    model = instantiate_model_from_checkpoint_folder_and_classname(checkpoint_folder)  # model already is in eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # try:
    #     model = torch.compile(model)
    # except Exception as err:
    #     print(err)

    # instantiate logger exactly as in training
    logdir = str(Path(checkpoint_folder).parents[2])
    model_name = os.path.basename(str(Path(checkpoint_folder).parents[1]))
    version = os.path.basename(str(Path(checkpoint_folder).parents[0]))
    logger = TensorBoardLogger(logdir, model_name, version=version)

    # create folder for all eval outputs
    output_folder = os.path.join(logdir, model_name, version, "eval")

    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    # save previous dataset state because we'll change it below
    previous_state = {
        'sequence_length': val_ds.sequence_length,
        'return_year': val_ds.return_year,
        'return_mode': val_ds.return_mode,
        'tree_ids': val_ds.tree_ids
    }

    # initialize dataloader for test dataset
    g = torch.Generator()
    g.manual_seed(0)
    test_dataloader = torch.utils.data.DataLoader(
        val_ds,
        pin_memory=True,
        batch_size=256,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker
    )

    metrics = {}
    # years = (2018, 2020, 2022)
    # for seq_len in (64, 128):
    for seq_len in (64,):
        print(f"Validating S2GNFI for seq_len={seq_len}")
        t0 = time.time()
        accs_ds = []
        for year in val_years:
            val_ds.sequence_length = seq_len
            val_ds.return_year = year
            val_ds.return_mode = return_mode

            # select only trees that actually have data present in the selected year
            valid_tree_ids = val_ds.df.loc[val_ds.df['year'] == year, 'tree_id'].unique()
            val_ds.tree_ids = valid_tree_ids

            acc_ds, cm, val_pred = model.test_on_dataloader(test_dataloader)
            accs_ds.append(acc_ds)

            # plot_confusion_matrices(
            #     os.path.join(output_folder, "plots"), cm, data.classes, "S2GNFI", f"seq_len={seq_len}_year={year}")
            utils.save_pandas_as_sqlite(
                os.path.join(output_folder, f"prediction_seq_len={seq_len}_year={year}.sqlite"),
                [val_pred], ["val"], overwrite=True)

            np.savetxt(os.path.join(output_folder, f"cm_S2GNFI_seq_len={seq_len}.csv"), cm, delimiter=",")

        t1 = time.time()
        print(f"Validating S2GNFI took {t1-t0}s.")
        print(f"Validating exploratories for seq_len={seq_len}")

        acc_expl, kl_div_expl = validate_exploratories(
            model,
            exploratories_dir="/data_hdd/exploratories/S2_L2",
            class_mapping=dataconfig["class_mapping"],
            outpath=output_folder,
            time_encoding=dataconfig["time_encoding"],
            seq_len=seq_len,
            qai=dataconfig["quality_mask"],
            mean=dataconfig["mean"],
            stddev=dataconfig["stddev"],
            return_mode=return_mode,
            append_ndvi=dataconfig.get("append_ndvi", False)
        )

        t2 = time.time()
        print(f"Validating exploratories took {t2-t1}s.")
        print(f"Validating TreeSatAI")

        if os.path.exists("/data_local_ssd/treesat/validation_treesat"):
            treesat_dir = "/data_local_ssd/treesat/validation_treesat"
        else:
            treesat_dir = "/data_ssd/treesat/validation_treesat"

        acc_treesat, cm_treesat, kl_div_treesat, per_plot_res = validate_treesat(
            model,
            treesat_dir=treesat_dir,
            class_mapping=dataconfig["class_mapping"],
            time_encoding=dataconfig["time_encoding"],
            return_mode=return_mode,
            seq_len=seq_len,
            qai=dataconfig["quality_mask"],
            mean=dataconfig["mean"],
            stddev=dataconfig["stddev"],
            append_ndvi=dataconfig.get("append_ndvi", False)
        )

        t3 = time.time()
        print(f"Validating TreeSatAI took {t3-t2}s.")
        print(f"Total validation time {t3-t0}s.")

        with open(os.path.join(output_folder, f"treesat_kl_divs_seq_len={seq_len}.txt"), "w") as f:
            for (folder, kl_div, share_pred) in per_plot_res:
                # avoid \n in string representation of array...
                share_pred_str = ', '.join(map(str, list(share_pred)))
                f.write(f"{folder}; {kl_div}; [{share_pred_str}]\n")

        np.savetxt(os.path.join(output_folder, f"cm_treesat_seq_len={seq_len}.csv"), cm_treesat, delimiter=",")

        # plot_confusion_matrices(
        #     os.path.join(output_folder, "plots"), cm_treesat, data.classes, "treesat", f"seq_len={seq_len}")

        metrics_this_seq_len = {
            f"acc_ds_avg_seq_len={seq_len}": np.mean(accs_ds),
            f"acc_expl_seq_len={seq_len}": acc_expl,
            f"acc_treesat_seq_len={seq_len}": acc_treesat,
            f"kl_div_expl_seq_len={seq_len}": kl_div_expl,
            f"kl_div_treesat_seq_len={seq_len}": kl_div_treesat
        }

        metrics = metrics | metrics_this_seq_len

        write_report(output_folder, f"seq_len={seq_len}", val_years, metrics | {f"acc_ds_seq_len={seq_len}": accs_ds})

    # Restore the previous state
    for key, value in previous_state.items():
        setattr(val_ds, key, value)

    # write out results
    hyper_params = model_init_args | {"model": utils.classname(model)} | dataconfig
    logger.log_hyperparams(hyper_params, metrics=metrics)
    logger.finalize("success")

    tend = time.time()
    print(f"Total validation time: {tend-tstart}")
    return metrics


def train_and_validate(model_config, data, dataconfig, logdir, experiment_name, version, model_extra_args={},
                       trainer_extra_args={}, experiment_file="", do_validation=True, val_return_mode="single",
                       val_years=(2018, 2020, 2022)):
    """Wrapper function to train and validate the model."""
    model, output_folder, _, _, init_args = train(
        model_config=model_config,
        data=data,
        logdir=logdir,
        experiment_name=experiment_name,
        version=version,
        model_extra_args=model_extra_args,
        trainer_extra_args=trainer_extra_args,
        experiment_file=experiment_file
    )

    save_classes(output_folder, data.classes)
    utils.save_dict_to_yaml({"model": {"class_path": str(model.__class__), "init_args": init_args}, "data": dataconfig},
                            os.path.join(output_folder, "config.yaml"))

    if do_validation:
        metrics = validate(
            checkpoint_folder=os.path.join(output_folder, "checkpoints"),
            val_ds=data.val_data,
            return_mode=val_return_mode,
            val_years=val_years,
            num_workers=len(os.sched_getaffinity(0)) // 2
        )
    else:
        metrics = {}

    return metrics
