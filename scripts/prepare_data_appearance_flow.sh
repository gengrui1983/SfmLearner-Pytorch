#!/usr/bin/env bash

raw_dataset=./../data/KITTI_origin
formatted_dataset=./../data/KITTI_formatted_appearance_flow

python3 data/prepare_train_data.py ${raw_dataset} --dataset-format 'kitti' --dump-root ${formatted_dataset} --width 224 --height 224 --num-threads 4 --static-frames ./data/static_frames.txt --with-depth --with-pose