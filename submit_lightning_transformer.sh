#! /bin/bash
#SBATCH -w node5
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.out

/opt/miniconda/bin/activate trex

#srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_10_classes.yaml \
#--model.hidden_dim 64 --model.transformer_layercount 5 --model.num_attention_heads 4 \
#--trainer.logger.init_args.name transformer \
#--data.quality_mask 223

srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_10_classes.yaml \
--model.hidden_dim 128 --model.transformer_layercount 8 --model.num_attention_heads 8 \
--trainer.logger.init_args.name transformer \
--data.quality_mask 223