#!/bin/bash

cd ..

export PYTHONPATH="/userhome/tools/zsl_pkg":$PYTHONPATH


MODEL=momn
DATA=cub
BACKBONE=resnet101
SAVE_PATH=./${DATA}/checkpoints/${MODEL}

mkdir -p ${SAVE_PATH}

nvidia-smi
python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 128 --lr 0.1 --resize_size 560 --crop_size 512 --epochs 90 --is_fix --pretrained &> ${SAVE_PATH}/fix.log
python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 16 --lr 0.001 --resize_size 560 --crop_size 512 --epochs 180 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.log
