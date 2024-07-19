#! /bin/bash
#SBATCH -w node5
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.out

source /opt/miniconda/etc/profile.d/conda.sh
conda activate trex

# Transformer
srun python ../main_CLI.py \
-c ./configs/transformer.yaml \
-c ./configs/14_classes.yaml \
-c ./configs/validation.yaml \
--trainer.max_steps 30000 \
--trainer.precision bf16-mixed \
--data.num_workers $NUM_WORKERS \
--data.pos_encode "absolute" \
--model.cosine_init_period 30001