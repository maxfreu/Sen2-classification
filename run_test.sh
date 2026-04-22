#! /bin/bash
DIR=/home/max/dr/Sen2-classification
pixi run python inference.py \
-i "${DIR}/testdata/X0063_Y0060" \
-o "${DIR}/testoutput" \
--checkpoint "${DIR}/output/final_cross_validation_GRU/cross_val=0/checkpoints/epoch=38-step=99411-val_loss=0.7419.ckpt" \
--model-config "${DIR}/output/final_cross_validation_GRU/cross_val=0/config.yaml" \
--data-config "${DIR}/output/final_cross_validation_GRU/cross_val=0/config.yaml" \
--qai 31 \
--sequence-length 64 \
--tmin-data 2025-01-01 \
--tmax-data 2026-01-01 \
--soft \
--num-classes 14 \
--device cpu \
--verbose