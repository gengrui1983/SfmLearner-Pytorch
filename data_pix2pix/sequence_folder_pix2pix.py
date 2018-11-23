import os
import random

import numpy as np
import torch.utils.data as data
from imageio import imread
from path import Path


def load_as_float(path):
    return np.asarray(imread(path).astype(np.float32))


class SequenceFolderPix2pix(data.Dataset):

    def __init__(self, root, segmentation, boundary, seed=None, train=True, sequence_length=2, transform=None,
                 target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.segmentation = Path(segmentation)
        self.boundary = Path(boundary)
        scene_list_path = self.root / 'train.txt' if train else self.root / 'val.txt'
        print(os.path.abspath("."), self.root)
        self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.seg_scenes = [self.segmentation / folder[:-1] for folder in open(scene_list_path)]
        self.boundary_scenes = [self.boundary / folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.samples = None
        # self.crawl_folders(sequence_length)
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        print("crawling...")
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene, seg_scene, boundary_scene in zip(self.scenes, self.seg_scenes, self.boundary_scenes):
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            segmentations = sorted(seg_scene.files('*.png'))
            boundaries = sorted(boundary_scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                          'seg': segmentations[i], 'ref_seg': [], 'boundary': boundaries[i], 'ref_boundaries': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                    sample['ref_seg'].append(segmentations[i + j])
                    sample['ref_boundaries'].append(boundaries[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        print("length:", len(sequence_set))
