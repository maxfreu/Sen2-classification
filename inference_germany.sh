#! /bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=kisski
#SBATCH --account=kisski-hawk-s2treeinference
#SBATCH --output=slurmlogs/slurm-%A-%a.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --nice=1
#SBATCH --array=0-515:8


module load rclone
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate inference

#YEAR=2022
NEXT_YEAR=$((YEAR + 1))
#MODEL_ID=2

MODEL_FOLDER="$HOME/models_doy"
CHECKPOINT="${MODEL_FOLDER}/gru_$MODEL_ID.ckpt"
MODEL_CONFIG="${MODEL_FOLDER}/config_noids.yaml"

OUTPUT_FOLDER="$PROJECT/maps_doy_time_encoding/${YEAR}_${MODEL_ID}"
mkdir -p $OUTPUT_FOLDER

# Get the list of all subdirectories
FILES=($(cat ~/filelist.txt | grep _$YEAR.tar | grep -v index))

process_file() {
    FILE_ID=$1
    GPU_ID=$2

    if [ $((SLURM_ARRAY_TASK_ID + GPU_ID)) -gt 515 ]; then
        return 0
    fi

    # Get the subfolder corresponding to this array job and the GPU within the node
    FILE=${FILES[$FILE_ID]}
    TILE=$(dirname $FILE)
    FILENAME=$(basename $FILE)

    if [ -f "$OUTPUT_FOLDER/$TILE.tif" ]; then
        echo "Skipping existing file $OUTPUT_FOLDER/$TILE.tif"
        return 0
    fi

    INPUT_FOLDER="$LOCAL_TMPDIR/$TILE"

    echo "Input folder: $INPUT_FOLDER"
    mkdir $INPUT_FOLDER

    echo "Source folder: $HOME/.project/dir.project/FORCE_GER/$TILE/$FILENAME"

    # copy stuff to node RAM
    # tar -xf $FILE -C $INPUT_FOLDER
    # rclone cat "s3-force:forst-sentinel2/FORCE_GER/$FILE" | tar -xf - -C $INPUT_FOLDER
    tar xf "$HOME/.project/dir.project/FORCE_GER/$TILE/$FILENAME" -C $INPUT_FOLDER

    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract tile $TILE/$FILENAME"
        return 0
    fi

    # Check if the subfolder exists
    if [ -d "$INPUT_FOLDER" ]; then
        echo "Processing $INPUT_FOLDER"
        # srun --overlap --export=ALL,CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py
        CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py \
        -i "$INPUT_FOLDER" \
        -o $OUTPUT_FOLDER \
        --checkpoint $CHECKPOINT \
        --model-config $MODEL_CONFIG \
        --data-config $MODEL_CONFIG \
        --qai 31 \
        --sequence-length 64 \
        --tmin-data "$YEAR-01-01" \
        --tmax-data "$NEXT_YEAR-01-01" \
        --soft \
        --num-classes 14
    else
        echo "Subfolder $INPUT_FOLDER does not exist."
    fi

    rm -rf $INPUT_FOLDER
}

process_file $((SLURM_ARRAY_TASK_ID + 0)) 0 &
process_file $((SLURM_ARRAY_TASK_ID + 1)) 0 &
process_file $((SLURM_ARRAY_TASK_ID + 2)) 1 &
process_file $((SLURM_ARRAY_TASK_ID + 3)) 1 &
process_file $((SLURM_ARRAY_TASK_ID + 4)) 2 &
process_file $((SLURM_ARRAY_TASK_ID + 5)) 2 &
process_file $((SLURM_ARRAY_TASK_ID + 6)) 3 &
process_file $((SLURM_ARRAY_TASK_ID + 7)) 3 &

wait

