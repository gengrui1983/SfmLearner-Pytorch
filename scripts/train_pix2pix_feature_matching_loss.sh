#!/usr/bin/env bash

data=../data/KITTI_formatted
segmentation=../data/KITTI_result
name=pix2pix-multiscale-feature_matching_loss

python train_p2p.py --data ${data} --segmentation ${segmentation} --save_path ${name} --name ${name} \
    --model pix2pix --which_model_netG resnet_6blocks_mlp \
    --which_model_netD multi_scale --which_direction AtoB --lambda_A 100 \
    --dataset_mode aligned --norm batch --pool_size 0 --num_D 3 --seed 100 --epoch_size 3000 \
    --input_nc 6 --output_nc 3