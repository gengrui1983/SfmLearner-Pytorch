#!/usr/bin/env bash

formatted_dataset=./../data/KITTI_formatted

python3 train.py ${formatted_dataset} -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output