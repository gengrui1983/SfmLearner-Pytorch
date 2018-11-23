#!/usr/bin/env bash

pose_path=../checkpoints/KITTI_formatted,epoch_size3000,m0.2/11-07-22:40/exp_pose_model_best.pth.tar
img_path=./../data/KITTI_SEGMENTATION/2011_09_26_drive_0005_sync_03

python3 test_pose.py ${pose_path} --dataset-dir ${img_path} --sequences [09]