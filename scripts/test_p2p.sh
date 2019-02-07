#!/usr/bin/env bash

data=../data/KITTI_formatted
segmentation=../data/KITTI_result
name=pix2pix-multiscale-feature_matching_loss
depth=../data/KITTI_DEPTH_RESULT
save=../data/KITTI_final_results

python test_p2p.py \
    --data ${data} --segmentation ${segmentation} \
    --depth ${depth} \
    --name ${name} \
    --save_path ${name} \
    --model pix2pix \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --use_dist \
    --input_nc 7 \
    --how_many -1 \
    --results_dir ${save}