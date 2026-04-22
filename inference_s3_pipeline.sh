#! /bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=kisski
#SBATCH --account=kisski-hawk-s2treeinference
#SBATCH --output=slurmlogs/slurm-%A-%a.out
#SBATCH --error=slurmlogs/slurm-%A-%a.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --array=0-9%2
#SBATCH -C inet

set -euo pipefail

module load rclone
MAMBA=/sw/rev/24.05/rome_ib_cuda_rocky8/linux-rocky8-zen2/gcc-11.4.0/micromamba-1.4.2-w3tnq7cnq77sz7l5j55qapbv65h2anpe/bin/micromamba
eval "$($MAMBA shell hook --shell=bash)"
micromamba activate inference

YEAR=2025
NEXT_YEAR=$((YEAR + 1))

S3_PREFIX="s3-force:forst-sentinel2/force/L2/ard"

MODEL_FOLDER="$HOME/models_final"
CHECKPOINTS=($(find $MODEL_FOLDER -name "*.ckpt"))
MODEL_CONFIG="${MODEL_FOLDER}/config_noids.yaml"

OUTPUT_FOLDER="$PROJECT/maps_final/${YEAR}"

echo "$CHECKPOINTS"

# Get the list of all tile folders from S3
TILES=($(rclone lsd "$S3_PREFIX" | awk '{print $NF}'))

process_data () {
  GPU_ID=$1
  CUDA_VISIBLE_DEVICES="$GPU_ID" python inference_pipeline.py \
    --s3-prefix "$S3_PREFIX" \
    --tiles "${TILES[@]}" \
    --output-folder "$OUTPUT_FOLDER" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --config "$MODEL_CONFIG" \
    --qai 31 \
    --sequence-length 64 \
    --tmin-data "$YEAR-01-01" \
    --tmax-data "$NEXT_YEAR-01-01" \
    --soft \
    --num-classes 14 \
    --parallel-loaders 2 \
    --parallel-processors 2 \
    --queue-size 3 \
    --batch-size 3000 \
    --world-size $((SLURM_ARRAY_TASK_COUNT * 4)) \
    --rank $((SLURM_ARRAY_TASK_ID * 4 + GPU_ID))
}

process_data 0 &
process_data 1 &
process_data 2 &
process_data 3 &

wait
