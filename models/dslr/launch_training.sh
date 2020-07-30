#!/bin/bash

# Network hyperparameters
num_grad_steps=5
num_resblocks=2
num_features=256
num_basis=8
device=0

# Name of model
model_name=train-DSLR$((num_basis))_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features

# Set folder names
dir_data=/data/sandino/Cine
dir_summary=$dir_data/summary/$model_name

python3 train.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --num-grad-steps $num_grad_steps \
				 --num-resblocks $num_resblocks \
				 --num-features 128 \
				 --num_basis 8 \
				 --num-emaps 2 \
				 --slwin-init \
				 --overlapping \
				 --device-num $device 