#!/usr/bin/env sh

GPU=0
CONFIG=biaffine_scibert_scidtb

# Training
python main_arcfactored.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=...
# python main_arcfactored.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate

