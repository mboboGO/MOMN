#!/bin/bash


export PYTHONPATH="/userhome/tools/zsl_pkg":$PYTHONPATH


MODEL=momn
DATA=cub
BACKBONE=densenet201
SAVE_PATH=./${DATA}/checkpoints/${MODEL}_dense_89.54


python eval.py -a ${MODEL} -d ${DATA} --backbone ${BACKBONE} -b 128 --resize_size 560 --crop_size 512 --is_fix --resume ${SAVE_PATH}/momn_89.5409.model

