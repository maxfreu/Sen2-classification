#! /bin/bash
#SBATCH -w node5
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurmlogs/slurm-%j.out
#SBATCH --error=./slurmlogs/slurm-%j.out
###SBATCH --ntasks-per-gpu=1

source /opt/miniconda/etc/profile.d/conda.sh
conda activate trex

#srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_13_classes_pure.yaml \
#--trainer.logger.init_args.name transformer_small_13_classes_fast \
#--data.where "" \
#--model.use_weighted_loss True
#
#srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_13_classes_pure.yaml \
#--trainer.logger.init_args.name transformer_small_13_classes_fast_cont \
#--data.where "present_2022 = TRUE" \
#--model.use_weighted_loss True
#
#srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_deciduous_pure.yaml \
#--trainer.logger.init_args.name transformer_small_deciduous_fast \
#--data.where "species >= 100" \
#--model.use_weighted_loss True

#srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_deciduous_pure.yaml \
#--trainer.logger.init_args.name transformer_small_deciduous_fast_cont \
#--data.where "species >= 100 and present_2022 = TRUE" \
#--model.use_weighted_loss True

srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_coniferous_pure.yaml \
--trainer.logger.init_args.name transformer_small_coniferous_fast \
--data.where "0 < species and species < 100" \
--model.use_weighted_loss True

srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_coniferous_pure.yaml \
--trainer.logger.init_args.name transformer_small_coniferous_fast_cont \
--data.where "0 < species and species < 100 and present_2022 = TRUE" \
--model.use_weighted_loss True

srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_3_classes_pure.yaml \
--trainer.logger.init_args.name transformer_small_3_classes_fast \
--data.where "" \
--model.use_weighted_loss True

srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_3_classes_pure.yaml \
--trainer.logger.init_args.name transformer_small_3_classes_fast_cont \
--data.where "present_2022 = TRUE" \
--model.use_weighted_loss True

# srun python lightning_main_CLI_transformer.py -c ./configs/config_transformer_10_classes.yaml \
# --model.hidden_dim 128 --model.transformer_layercount 8 --model.num_attention_heads 8 \
# --trainer.logger.init_args.name transformer_large \
# --data.quality_mask 223
