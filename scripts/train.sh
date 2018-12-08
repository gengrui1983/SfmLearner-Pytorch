#!/usr/bin/env bash

formatted_dataset=./../data/KITTI_formatted
segmentation=./../data/KITTI_SEGMENTATION
boundary=./../data/KITTI_BOUNDARIES

python3 train.py ${formatted_dataset} --segmentation ${segmentation} -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output