#!/usr/bin/env sh

GPU=0
CONFIG=stackpointer_scibert_scidtb

# Training
python main_stackpointer.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --actiontype train

# Evaluation
# MYPREFIX=...
# python main_stackpointer.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --prefix ${MYPREFIX} \
#     --actiontype evaluate


