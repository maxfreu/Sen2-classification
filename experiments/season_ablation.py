import os

import yaml

from experiments.train_and_validate import load_data, train_and_validate


logdir = "output"
experiment_name = "season_ablation"
model_config = "/home/max/dr/Sen2-classification/configs/gru_best.yaml"
month_day = "strftime(to_timestamp(time), '%m-%d')"
season_conditions = {
    "spring": f"({month_day} BETWEEN '03-01' AND '05-31')",
    "summer": f"({month_day} BETWEEN '06-01' AND '08-31')",
    "autumn": f"({month_day} BETWEEN '09-01' AND '11-30')",
    "winter": f"(({month_day} BETWEEN '12-01' AND '12-31') OR ({month_day} BETWEEN '01-01' AND '02-29'))",
}


with open("configs/statistics_223_g-5k.yaml") as f:
    norm_config = yaml.safe_load(f)["data"]

runs = [None, "spring", "summer", "autumn", "winter"]
run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
omitted_season = runs[run_id]
where = "" if omitted_season is None else f"NOT {season_conditions[omitted_season]}"
version = "baseline_full_year" if omitted_season is None else f"without_{omitted_season}"

print(f"Running season ablation {run_id}: {version}")
print(f"where: {where or 'full year'}")

data, dataconfig = load_data(
    overwrite_args={
        "mean": norm_config["mean"],
        "stddev": norm_config["stddev"],
        "where": where,
        "val_where": where,
    }
)

train_and_validate(
    model_config,
    data,
    dataconfig | {"normalization": "223_g-5k", "omitted_season": omitted_season or "none"},
    logdir,
    experiment_name=experiment_name,
    version=version,
    experiment_file=__file__,
    model_extra_args={
        "num_classes": data.num_classes,
        "classes": data.classes,
    },
)
