#!/usr/bin/env bash

DISPNET_PATH=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/11-07-22:40/dispnet_model_best.pth.tar
POSTNET_PATH=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/11-07-22:40/exp_pose_model_best.pth.tar
DATA_PATH=./../data/KITTI_SEGMENTATION
DATA_LIST=./../data/KITTI_formatted/val.txt

python3 test_disp.py --pretrained-dispnet $DISPNET_PATH \
    --pretrained-posenet $POSTNET_PATH \
    --dataset-dir $DATA_PATH \
    --dataset-list $DATA_LIST