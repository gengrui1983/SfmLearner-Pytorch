#!/usr/bin/env bash

#formatted_dataset=./../data/KITTI_formatted
formatted_dataset=./../data/kitti_odemetry_224_224
#segmentation=./../data/KITTI_SEGMENTATION
segmentation=./../data/kitti_odemetry_segmented_images_colored_224_224
#boundary=./../data/KITTI_BOUNDARIES
depth=../data/KITTI_DEPTH_RESULT

python3 train_depth.py ${formatted_dataset} --depth ${depth} \
        --segmentation ${segmentation} -b4 -m0.2 -s0.1 -f500\
        --epoch-size 3000 --sequence-length 5 --log-output -j1 \
        --epochs 200 --lr 2e-5\
        --odemetry