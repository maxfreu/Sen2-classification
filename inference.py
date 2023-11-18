#! /bin/env python3
import argparse
from sen2classification.classifiers import SBERTClassifier

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--ckpt", default="/home/max/dr/Sen2-classification/output/transformer/version_1/checkpoints/epoch=27-step=22036.ckpt")
parser.add_argument("--qai", default=0, type=int, help="Quality assurance information integer flag")
parser.add_argument("-o", dest="outfile", default=None,
                    help="Output file path including name and extension. If none is given, a file <input folder>.tif will be crated in the input folder.")
args = parser.parse_args()

model = SBERTClassifier.load_from_checkpoint(args.ckpt, max_embedding_size=3225)
model = model.eval()
model = model.compile()
model = model.to("cuda")

model.predict_timeseries(args.dir,
                         output_filepath=args.outfile,
                         qai=args.qai,
                         seq_len=128,
                         batch_size=1000)

