#!/usr/bin/env sh

GPU=0
CONFIG=arcstandard_scibert_scidtb

# Training
python main_shiftreduce.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=...
# python main_shiftreduce.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate

