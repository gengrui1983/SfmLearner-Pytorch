#!/usr/bin/env bash

formatted_dataset=./../data/KITTI_formatted
segmentation=./../data/KITTI_SEGMENTATION
boundary=./../data/KITTI_BOUNDARIES
pretrained_exp_pose=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/01-25-23:58/exp_pose_model_best.pth.tar
pretrained_disp=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/01-25-23:58/dispnet_model_best.pth.tar
output_dir=../data/KITTI_result

python3 train.py ${formatted_dataset} --segmentation ${segmentation} --boundary ${boundary} -b2 -m0.2 -s0.1 \
--epoch-size 19000 --sequence-length 3 --log-output \
--pretrained-exppose ${pretrained_exp_pose} \
--pretrained-disp ${pretrained_disp} --output-dir ${output_dir} --evaluate