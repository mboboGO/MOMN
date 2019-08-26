# MOMN
This is the implementation of Multi-Objective Matrix Normalization for Fine-grained Visual Recognition by Pytorch.


# Requirements
pytorch 1.0

# Training
The training scripts for CUB, Cars, Air, and Dogs are given in https://drive.google.com/drive/folders/1mgKoXwDg3oUGiJluCSWlZJkvrhsbq2tw?usp=sharing.
Other extensions can be easily modified.

A detailed illustration is as follows:

step 1:
> adding your data path around the 130 line in main.py

step 2:
> creating a running bash scrip, as the given example in the google drive.
> specifically, the running command should be given by:
>> python main.py -a momn -d cub -s ./cub/checkpoints --backbone densenet201 -b 230 --lr 0.1 --resize_size 560 --crop_size 512 --epochs 90 --is_fix --pretrained

# Testing
An example script for testing is also given in the google drive.
