#!/usr/bin/env bash

#formatted_dataset=./../data/KITTI_formatted
formatted_dataset=./../data/kitti_odemetry_224_224
#segmentation=./../data/KITTI_SEGMENTATION
segmentation=./../data/kitti_odemetry_segmented_images_colored_224_224
#boundary=./../data/KITTI_BOUNDARIES
#pretrained_exp_pose=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/12-03-20:11/exp_pose_model_best.pth.tar
#pretrained_disp=./checkpoints/KITTI_formatted,epoch_size3000,m0.2/12-03-20:11/dispnet_model_best.pth.tar

python3 train.py ${formatted_dataset} \
        --segmentation ${segmentation} -b4 -m0.2 -s0.1 \
        --epoch-size 3000 --sequence-length 3 --log-output \
        --odemetry
#        --pretrained-exppose ${pretrained_exp_pose} \
#        --pretrained-disp ${pretrained_disp}