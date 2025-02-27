#! /bin/bash
#SBATCH -w node5
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.out

source /opt/miniconda/etc/profile.d/conda.sh
conda activate trex

# no data layernorm
# division by 10k

# GRU
srun python ../main_CLI.py \
-c ./configs/gru.yaml \
-c ./configs/14_classes.yaml \
-c ./configs/validation.yaml \
-c ./configs/statistics_10k.yaml \
--trainer.max_steps 30000 \
--trainer.precision bf16-mixed \
--trainer.logger.init_args.version nonorm_10k \
--data.num_workers $NUM_WORKERS \
--data.pos_encode "absolute" \
--model.cosine_init_period 30001 \

# Transformer
srun python ../main_CLI.py \
-c ./configs/transformer.yaml \
-c ./configs/14_classes.yaml \
-c ./configs/validation.yaml \
-c ./configs/statistics_10k.yaml \
--trainer.max_steps 30000 \
--trainer.precision bf16-mixed \
--trainer.logger.init_args.version nonorm_10k \
--data.num_workers $NUM_WORKERS \
--data.pos_encode "absolute" \
--model.cosine_init_period 30001
