#!/usr/bin/env sh

GPU=0

# Source-only (A)

CONFIG_SO_A=so_a_scidtb_cord19_covid19dtb

python main_tacl2022.py \
    --gpu ${GPU} \
    --config ${CONFIG_SO_A} \
    --actiontype train

# PREFIX_SO_A=...
# python main_tacl2022.py \
#     --gpu ${GPU} \
#     --config ${CONFIG_SO_A} \
#     --prefix ${PREFIX_SO_A} \
#     --actiontype evaluate

# Source-only (S)

CONFIG_SO_S=so_s_scidtb_cord19_covid19dtb

python main_tacl2022.py \
    --gpu ${GPU} \
    --config ${CONFIG_SO_S} \
    --actiontype train

# PREFIX_SO_S=...
# python main_tacl2022.py \
#     --gpu ${GPU} \
#     --config ${CONFIG_SO_S} \
#     --prefix ${PREFIX_SO_S} \
#     --actiontype evaluate

# Source-only (B)

CONFIG_SO_B=so_b_scidtb_cord19_covid19dtb

python main_tacl2022.py \
    --gpu ${GPU} \
    --config ${CONFIG_SO_B} \
    --actiontype train

# PREFIX_SO_B=...
# python main_tacl2022.py \
#     --gpu ${GPU} \
#     --config ${CONFIG_SO_B} \
#     --prefix ${PREFIX_SO_B} \
#     --actiontype evaluate

# Common

PRETRAINED_A=tacl2022.${CONFIG_SO_A}/${PREFIX_SO_A}.model.0
PRETRAINED_S=tacl2022.${CONFIG_SO_S}/${PREFIX_SO_S}.model.0
PRETRAINED_B=tacl2022.${CONFIG_SO_B}/${PREFIX_SO_B}.model.0

# Co-training (A, S)

CONFIG=ct_as_scidtb_cord19_covid19dtb_above06

python main_tacl2022.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --pretrained ${PRETRAINED_A} ${PRETRAINED_S} \
    --actiontype train

# MYPREFIX=...
# python main_tacl2022.py \
#     --gpu ${GPU} \
#     --config ${CONFIG} \
#     --pretrained ${PRETRAINED_A} ${PRETRAINED_S} \
#     --prefix ${MYPREIFX} \
#     --actiontype evaluate



