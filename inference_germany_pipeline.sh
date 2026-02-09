#! /bin/bash
#SBATCH --time=00:75:00
#SBATCH --partition=kisski
#SBATCH --account=kisski-hawk-s2treeinference
#SBATCH --output=slurmlogs/slurm-%A-%a.out
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --nice=100
#SBATCH --array=0-19


module load rclone
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate inference

YEAR=2022
NEXT_YEAR=$((YEAR + 1))
#MODEL_ID=2

MODEL_FOLDER="$HOME/models_final_final"
CHECKPOINTS=($(find $MODEL_FOLDER -name "*.ckpt"))
MODEL_CONFIG="${MODEL_FOLDER}/config_noids.yaml"

OUTPUT_FOLDER="$PROJECT/maps_final_final/${YEAR}"
#mkdir -p $OUTPUT_FOLDER

echo "$CHECKPOINTS"

# Get the list of all subdirectories
FILES=$(lfs find $PROJECT/FORCE_GER --name "*_$YEAR.tar")

#echo $MODEL_FOLDER
#echo $CHECKPOINT
#echo $OUTPUT_FOLDER
#echo $MODEL_CONFIG

process_data () {
  GPU_ID=$1
  CUDA_VISIBLE_DEVICES="$GPU_ID" python inference_pipeline.py \
    --tar-files $FILES \
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
