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

# args = parser.parse_args(["/home/max/dr/Sen2-classification/output/embedding_test_SBERTClassifier/embedding_type=bert/checkpoints"])
args = parser.parse_args()
config_path = checkpoint_folder_to_configfile(args.checkpoint_folder)

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
test_dataset = InMemoryTimeSeriesDataset(
    input_filepath=input_filepath,
    # input_filepath="/home/max/dr/extract_sentinel_pixels/datasets/S2GNFI_V1.parquet",
    dbname=dataconfig["dbname"],
    sequence_length=dataconfig["sequence_length"],
    quality_mask=dataconfig["quality_mask"],
    class_mapping=dataconfig["class_mapping"],
    return_mode=dataconfig["return_mode"],
    time_encoding=dataconfig["time_encoding"],
    plot_ids=dataconfig["plot_ids"] if "plot_ids" in dataconfig.keys() else None,
    where="" if "plot_ids" in dataconfig.keys() else "is_train = FALSE",
    # where="tree_id < -65000"
    mean=dataconfig["mean"],
    stddev=dataconfig["stddev"]
)

#%%
validate(checkpoint_folder=args.checkpoint_folder,
         val_ds=test_dataset,
         return_mode=dataconfig["return_mode"],
         num_workers=args.workers)
