#!/usr/bin/env bash

DATASET=~/Documents/Dev/data/KITTI_SEGMENTATION/
BOUNDARY_ROOT=~/Documents/Dev/data/KITTI_BOUNDARIES/

python ./tools/plot_canny.py $DATASET --dump-boundary-root $BOUNDARY_ROOT