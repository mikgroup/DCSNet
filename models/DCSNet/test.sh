#!/bin/bash

# Network hyperparameters
num_grad_steps=5
num_resblocks=2
num_features=64
device=1
modlflag=True
CG_steps=8
date=0704
batch_size=4
loss_type=2
loss_normalized=True
Network=ResUNet
# Name of model
model_name=train_3DMUDI_Network_$((Network))_data_$((date))_batchsize_$((batch_size))
# 3D_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))

# Set folder names
dir_data=/home/kewang/MUDI_data_train/
dir_summary=$dir_data/summary/$model_name

python3 train.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --device-num $device \
                 --batch-size $batch_size \
				 --loss-normalized $loss_normalized \
                 --sample-rate 0.1 \
# 				 --num-grad-steps $num_grad_steps \
# 				 --num-resblocks $num_resblocks \
# 				 --num-features $num_features \
# 				 --num-emaps 1 \
# 				 --slwin-init \

# 				 --loss-normalized $loss_normalized \
# 				 --loss-type $loss_type \