import os
from pathlib import Path

import yaml

from experiments.augmentation_study_runs import build_runs
from experiments.train_and_validate import load_data, train_and_validate
from sen2classification import utils


TRAIN_WHERE = "(time < 1609459200)"
VAL_WHERE = "(1609459200 <= time and time < 1672531200 and present_2022=1)"
VAL_YEARS = (2021, 2022)
RETURN_MODE = "single"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    git_commit_hash = utils.get_git_hash_safe(repo_root, ignore_untracked=True)

    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    experiment_name = os.environ.get("AUGMENTATION_STUDY_NAME", "augmentation_study")
    logdir = os.environ.get("AUGMENTATION_STUDY_LOGDIR", "output")
    model_config = os.environ.get("AUGMENTATION_STUDY_MODEL_CONFIG", "configs/gru_best.yaml")
    data_config = os.environ.get("AUGMENTATION_STUDY_DATA_CONFIG", "configs/14_classes.yaml")
    normalization_config = os.environ.get(
        "AUGMENTATION_STUDY_NORMALIZATION_CONFIG",
        "configs/statistics_223_g-5k.yaml",
    )
    max_epochs = os.environ.get("AUGMENTATION_STUDY_MAX_EPOCHS")
    resume_from_checkpoint = os.environ.get("AUGMENTATION_STUDY_RESUME_FROM")

    with open(normalization_config) as f:
        norm_config = yaml.safe_load(f)["data"]

    runs = build_runs()

    if run_id < 0 or run_id >= len(runs):
        raise IndexError(f"Run index {run_id} is out of range for {len(runs)} augmentation runs.")

    run = runs[run_id]

    print(f"Running augmentation study {run_id + 1}/{len(runs)}: {run['label']}")
    print(yaml.safe_dump(run, sort_keys=False))
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    data, dataconfig = load_data(
        dataconfigfile=data_config,
        overwrite_args={
            "return_mode": RETURN_MODE,
            "mean": norm_config["mean"],
            "stddev": norm_config["stddev"],
            "where": TRAIN_WHERE,
            "val_where": VAL_WHERE,
            "augmentation_kwargs": run["augmentation_kwargs"],
        },
    )

    version = f"{run_id:02d}_{run['name']}"

    trainer_extra_args = {}
    if max_epochs is not None:
        trainer_extra_args["max_epochs"] = int(max_epochs)

    train_and_validate(
        model_config,
        data,
        dataconfig
        | {
            "git_commit_hash": git_commit_hash,
            "normalization": "223_g-5k",
            "augmentation_study_run": {
                "id": run_id,
                "name": run["name"],
                "label": run["label"],
                "group": run["group"],
            },
        },
        logdir,
        experiment_name=experiment_name,
        version=version,
        experiment_file=__file__,
        trainer_extra_args=trainer_extra_args,
        model_extra_args={
            "num_classes": data.num_classes,
            "classes": data.classes,
        },
        val_return_mode=RETURN_MODE,
        val_years=VAL_YEARS,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()