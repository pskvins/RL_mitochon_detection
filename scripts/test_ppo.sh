#!/bin/bash
#SBATCH --job-name=ppo_test
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --time=13-11:30:00 # d-hh:mm:ss Çü½Ä, º»ÀÎ jobÀÇ max time limit ÁöÁ¤ 
#SBATCH --mem=30000 # cpu memory size 
#SBATCH --cpus-per-task=4 # cpu °³¼ö 
#SBATCH -o ./logs/ppo_test.txt
#SBATCH -e ./logs/ppo_test.err

ml purge
ml load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate smd

cd /data2/projects/seunghoon/courseworks/25-RL/RL_mitochon_detection/


EXP_DIR=/data2/projects/seunghoon/courseworks/25-RL/RL_mitochon_detection/
# constants
CONFIG=src/ppo_model/configs/config.yaml
CKPT_DIR=ppo_runs/ppo_exp1/weights
LOG_DIR=ppo_runs/ppo_exp1/logs
EXP_NAME=ppo_test

# VAL_LMDB=datasets/lmdb/refcocog_u/val.lmdb
# TEST_LMDB=datasets/lmdb/refcocog_u/test.lmdb
# # variables
# HN_PROB=0.0
# TRAIN_TEXT_ENCODER=True
# TRAIN_VISUAL_ENCODER=False
# TRAIN_METRIC_LEARNING=True
# METRIC_MODE=hardpos_only_refined_sbertsim
# LOSS_OPTION=ACE_verbonly
# MARGIN_VAL=12
# EXP_NAME=only_add_hp_4
# OPT_DIR=exp/refcocog_u/refcocog_u_ACE_abl
# LOG_NAME=only_add_hp_4.log

# # core args
# BATCH_SIZE=64
# LOCALHOST=7025

# Create the directory if it does not exist
if [[ ! -d "${CKPT_DIR}/${EXP_NAME}" ]]; then
    echo "Directory ${CKPT_DIR}/${EXP_NAME} does not exist. Creating it..."
    mkdir -p "${CKPT_DIR}/${EXP_NAME}"
fi


python -u src/train_ppo.py --config "${CONFIG}"

    # --hn_prob "${HN_PROB}" \
    # --train_text_encoder "${TRAIN_TEXT_ENCODER}" \
    # --train_visual_encoder "${TRAIN_VISUAL_ENCODER}" \
    # --train_metric_learning "${TRAIN_METRIC_LEARNING}" \
    # --metric_mode "${METRIC_MODE}" \
    # --loss_option "${LOSS_OPTION}" \
    # --margin_val "${MARGIN_VAL}"