#!/usr/bin/env bash

#formatted_dataset=./../data/KITTI_formatted
formatted_dataset=./../data/kitti_odemetry_224_224
#segmentation=./../data/KITTI_SEGMENTATION
segmentation=./../data/kitti_odemetry_segmented_images_colored_224_224
boundary=./../data/KITTI_BOUNDARIES
pretrained_exp_pose=./checkpoints/kitti_odemetry_224_224,epoch_size3000,m0.2/05-18-23:44/exp_pose_model_best.pth.tar
pretrained_disp=./checkpoints/kitti_odemetry_224_224,epoch_size3000,m0.2/05-18-23:44/dispnet_model_best.pth.tar
output_dir=../data/KITTI_odemetry_result



python3 train.py ${formatted_dataset} --segmentation ${segmentation} --boundary ${boundary} -b2 -m0.2 -s0.1 \
--epoch-size 19000 --sequence-length 3 --log-output \
--pretrained-exppose ${pretrained_exp_pose} \
--pretrained-disp ${pretrained_disp} --output-dir ${output_dir} --evaluate