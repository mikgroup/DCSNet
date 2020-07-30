#!/bin/bash

# Network hyperparameters
device=3
date=7271
batch_size=3
loss_normalized=True
Network=UNet3D
Loss=perceptual_01
# Name of model
model_name=train_DCSNet_${Network}_data_$((date))_batchsize_$((batch_size))_${Loss}
# 3D_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))

# Set folder names
dir_data=/home/kewang/DCSNet/DCSNet_data/
dir_summary=$dir_data/summary/$model_name

python3 train.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --device-num $device \
                 --batch-size $batch_size \
				 --loss-normalized \
                 --conditional \
                 --vgg-perceptual \
                 --vgg-weight 0.1 \
#                  --adv \
#                  --gan-weight 0.1 \
# 				 --num-grad-steps $num_grad_steps \
# 				 --num-resblocks $num_resblocks \
# 				 --num-features $num_features \
# 				 --num-emaps 1 \
# 				 --slwin-init \

# 				 --loss-normalized $loss_normalized \
# 				 --loss-type $loss_type \