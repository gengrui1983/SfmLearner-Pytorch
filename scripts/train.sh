#!/usr/bin/env bash

#formatted_dataset=./../data/KITTI_formatted
formatted_dataset=./../data/kitti_odemetry_224_224
#segmentation=./../data/KITTI_SEGMENTATION
segmentation=./../data/kitti_odemetry_segmented_images_colored_224_224
#boundary=./../data/KITTI_BOUNDARIES
pretrained_exppose=./checkpoints/kitti_odemetry_224_224,epoch_size3000,m0.2/06-07-22:12/exp_pose_checkpoint.pth.tar
pretrained_disp=./checkpoints/kitti_odemetry_224_224,epoch_size3000,m0.2/06-07-22:12/dispnet_checkpoint.pth.tar

python3 train.py ${formatted_dataset} \
        --segmentation ${segmentation} -b4 -m0.2 -s0.1\
        --epoch-size 3000 --sequence-length 3 --log-output \
        --epochs 30 --odemetry
        #--pretrained-exppose ${pretrained_exppose} \
        #--pretrained-disp ${pretrained_disp} \

