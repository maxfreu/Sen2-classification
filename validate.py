import os
import yaml
from argparse import ArgumentParser
from experiments.train_and_validate import validate
from experiments.validation.val_utils import checkpoint_folder_to_configfile
from sen2classification.datasets import InMemoryTimeSeriesDataset


parser = ArgumentParser()
parser.add_argument("checkpoint_folder")
parser.add_argument("--workers", default=0, type=int)
# parser.add_argument("--return-mode", default="single", help="'single', 'double' or 'random'")

# args = parser.parse_args(["/home/max/dr/Sen2-classification/output/time_extrapolation_GRU/time_encoding=absolute-return_mode=single-num_layers=2-hidden_dim=128-fc_size=128-embedding_type=bert/checkpoints"])
args = parser.parse_args()
config_path = checkpoint_folder_to_configfile(args.checkpoint_folder)

print(f"Validating best model in: {args.checkpoint_folder}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

dataconfig = config["data"]

#%%
if os.path.exists(dataconfig["input_file"]):
    input_filepath = dataconfig["input_file"]
elif os.path.exists("/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet"):
    input_filepath = "/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet"
else:
    raise ValueError("Dataset file not found!")

#%%
where = dataconfig.get("where", "")
val_where = dataconfig.get("val_where", where)
val_where = f"({val_where}) AND is_train = FALSE" if val_where else "is_train = FALSE"

print(f"Where statement used for validation: {val_where}")

#%%
test_dataset = InMemoryTimeSeriesDataset(
    input_filepath=input_filepath,
    # input_filepath="/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet",
    dbname=dataconfig["dbname"],
    sequence_length=dataconfig["sequence_length"],
    quality_mask=dataconfig["quality_mask"],
    class_mapping=dataconfig["class_mapping"],
    return_mode=dataconfig["return_mode"],
    time_encoding=dataconfig["time_encoding"],
    plot_ids=dataconfig.get("val_ids"),
    where=val_where,
    mean=dataconfig["mean"],
    stddev=dataconfig["stddev"]
)

#%%
validate(checkpoint_folder=args.checkpoint_folder,
         val_ds=test_dataset,
         return_mode=dataconfig["return_mode"],
         num_workers=args.workers,
         val_years=(2020, 2021, 2022))
