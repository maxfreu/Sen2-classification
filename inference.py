import os.path
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from sen2classification import utils
from datetime import date


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--input-folder", "-i", dest="input_folder", type=str)
    parser.add_argument("--output-folder", "-o", dest="output_folder", type=str)
    parser.add_argument("--checkpoint", dest="checkpoint", type=str)
    parser.add_argument("--model-config", dest="model_config", type=str)
    parser.add_argument("--data-config",  dest="data_config", type=str)
    parser.add_argument("--qai", dest="qai", type=int)
    parser.add_argument("--sequence-length", dest="sequence_length", type=int)
    parser.add_argument("--tmin-data", dest="tmin_data", type=str,
                        help="Starting time for loading data. Format: yyyy-mm-dd")
    parser.add_argument("--tmax-data", dest="tmax_data", type=str,
                        help="Ending time for loading data (exclusive). Format: yyyy-mm-dd")
    parser.add_argument("--tmin-inference", dest="tmin_inference", type=str, default="",
                        help="Starting time for inference. Format: yyyy-mm-dd. Default: tmin_data")
    parser.add_argument("--tmax-inference", dest="tmax_inference", type=str, default="",
                        help="Ending time for inference (exclusive). Format: yyyy-mm-dd. Default: tmax_data")
    parser.add_argument("--soft", action="store_true",
                        help="If given, the output will be a multiband raster where each band represents the probability"
                        "for a given class, scaled to 0-255.")
    parser.add_argument("--num-classes", default=0, type=int, help="Number of classes the network outputs. "
                        "Only required, if --soft is given.")
    return parser


def main():
    # MODEL_FOLDER = "/home/max/dr/Sen2-classification/output/embedding_test_SBERTClassifier/embedding_type=concat"
    # MODEL_FOLDER = "/home/max/dr/Sen2-classification/output/time_encoding_GRU/time_encoding=doy-return_mode=single"
    # CHECKPOINT = f"{MODEL_FOLDER}/checkpoints/epoch=23-step=53256-val_loss=0.7466.ckpt"
    # MODEL_FOLDER = "/home/max/dr/Sen2-classification/output/embedding_test_SBERTClassifier/embedding_type=concat"
    # CHECKPOINT = f"{MODEL_FOLDER}/checkpoints/epoch=16-step=37723-val_loss=0.7514.ckpt"
    # MODEL_CONFIG = f"{MODEL_FOLDER}/config.yaml"
    # DATA_CONFIG  = f"{MODEL_FOLDER}/config.yaml"
    # SUBFOLDER = "/data_hdd/force_codede/FORCE/C1/L2/ard/X0061_Y0046"

    parser = get_parser()
    args = parser.parse_args()
    # args = parser.parse_args(f"--soft --num-classes 14 -i {SUBFOLDER} -o /data_hdd/force_codede/output/ --checkpoint {CHECKPOINT} --model-config {MODEL_CONFIG} --data-config {DATA_CONFIG} --qai 31 --sequence-length 128 --tmin-data 2017-01-01 --tmax-data 2019-01-01".split())

    if args.soft and args.num_classes == 0:
        raise parser.error("The argument --num-classes must be specified and greater 0 when --soft is given.")

    input_folder = args.input_folder
    output_folder = args.output_folder
    ckpt = args.checkpoint
    model_config_path = args.model_config
    data_config_path = args.data_config
    qai = args.qai
    sequence_length = args.sequence_length
    tmin_data = date(*[int(x) for x in args.tmin_data.split("-")])
    tmax_data = date(*[int(x) for x in args.tmax_data.split("-")])
    tmin_inference = date(*[int(x) for x in args.tmin_inference.split("-")]) if args.tmin_inference else tmin_data
    tmax_inference = date(*[int(x) for x in args.tmax_inference.split("-")]) if args.tmax_inference else tmax_data

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)["data"]
        mean   = np.array(data_config["mean"]).astype(np.float32)
        stddev = np.array(data_config["stddev"]).astype(np.float32)

    model, _ = utils.load_model_from_configs_and_checkpoint(model_config_path, data_config_path, ckpt)
    model.to("cuda")
    model.eval()
    model = torch.compile(model)

    output_filepath = os.path.join(output_folder, f"{os.path.basename(input_folder)}.tif")
    model.predict_force_folder(input_folder,
                               seq_len=sequence_length,
                               qai=qai,
                               output_filepath=output_filepath,
                               verbose=False,
                               time_encoding=data_config["time_encoding"],
                               mean=mean,
                               stddev=stddev,
                               batch_size=2048,
                               apply_argmax=not args.soft,
                               num_classes=args.num_classes,
                               # TODO: Remove this hardcoding!!!
                               band_reordering=(3,0,1,2,4,5,6,7,8,9,10,11,12,13),
                               tmin_data=tmin_data,
                               tmax_data=tmax_data,
                               tmin_inference=tmin_inference,
                               tmax_inference=tmax_inference,
                               append_ndvi=data_config.get("append_ndvi", False)
                               )

#%%
main()
