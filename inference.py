#! /bin/env python3
import argparse
import torch
from sen2classification.models.classifiers import SBERTClassifier
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--ckpt", default="/home/max/dr/Sen2-classification/output/transformer/version_3/checkpoints/epoch=28-step=22823.ckpt")
parser.add_argument("--qai", default=0, type=int, help="Quality assurance information integer flag")
parser.add_argument("--max-embed", default=2929, type=int, help="Maximum temporal embidding length")
parser.add_argument("--device", default="cpu", type=str, help="Computing device, cpu or cuda")
parser.add_argument("-o", dest="outfile", default=None,
                    help="Output file path including name and extension. If none is given, a file <input folder>.tif will be crated in the input folder.")
args = parser.parse_args()

model = SBERTClassifier.load_from_checkpoint(args.ckpt, max_embedding_size=args.max_embed, map_location=args.device)
model = model.eval()
model = torch.compile(model)

model.predict_timeseries(args.dir,
                         output_filepath=args.outfile,
                         qai=args.qai,
                         seq_len=128,
                         batch_size=1000,
                         t0=datetime.date(2016, 1, 1))

