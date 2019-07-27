#!/usr/bin/env bash

#formatted_dataset=./../data/KITTI_formatted
formatted_dataset=./../data/kitti_odemetry_224_224
#segmentation=./../data/KITTI_SEGMENTATION
segmentation=./../data/kitti_odemetry_segmented_images_colored_224_224
pretrained_exp_pose=./checkpoints/kitti_odemetry_224_224,30epochs,epoch_size3000,m0.2/06-08-22:00/exp_pose_model_best.pth.tar
pretrained_disp=./checkpoints/kitti_odemetry_224_224,30epochs,epoch_size3000,m0.2/06-08-22:00/dispnet_checkpoint.pth.tar
output_dir=../data/KITTI_odemetry_result

python3 train.py ${formatted_dataset} --segmentation ${segmentation} -b2 -m0.2 -s0.1 \
--epoch-size 19000 --sequence-length 3 --log-output \
--pretrained-exppose ${pretrained_exp_pose} \
--pretrained-disp ${pretrained_disp} --output-dir ${output_dir} --evaluate