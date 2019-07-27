import os
import random

import numpy as np
import torch
import torch.utils.data as data
from imageio import imread
from path import Path
import ipdb

from util.util import read_calib_file, read_transformations


def load_as_float(path):
    return np.asarray(imread(path).astype(np.float32))


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, segmentation, depth=None, seed=None, train=True, sequence_length=2, transform=None,
                 target_transform=None, pix2pix=False, p2p_benchmark=False, odemetry=False, depth_prediction=False):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.segmentation = Path(segmentation)
        self.p2p_benchmark = p2p_benchmark
        if pix2pix or depth_prediction:
            self.depth = Path(depth)

        if depth_prediction:
            scene_list_path = self.root / 'train_pix2pix.txt' if train else self.root / 'val_pix2pix.txt'
        elif not pix2pix:
            # scene_list_path = self.root / 'train.txt' if train else self.root / 'train_pix2pix.txt'
            scene_list_path = self.root / 'train_sfm.txt' if train else self.root / 'val_sfm.txt'
        else:
            scene_list_path = self.root / 'train_pix2pix.txt' if train else self.root / 'val_pix2pix.txt'

        print(os.path.abspath("."), self.root)
        self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
        self.seg_scenes = [self.segmentation / folder[:-1] for folder in open(scene_list_path)]

        if pix2pix or depth_prediction:
            self.seg_scenes = [self.segmentation / folder[:-1] for folder in open(scene_list_path)]
            self.depth_scenes = [self.depth / folder[:-1] for folder in open(scene_list_path)]

        self.transform = transform

        self.samples = None
        self.pix2pix = pix2pix
        self.depth_pred = depth_prediction
        # self.crawl_folders(sequence_length)
        if depth_prediction:
            self.craw_folders_depth_seg(sequence_length, train)
        elif pix2pix:
            self.crawl_folders_pix2pix()
        else:
            self.crawl_folders(sequence_length, train)

    def craw_folders_depth_seg(self, sequence_length, is_train=True):
        print('crawling...')
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        offset = 1

        shifts = list(range(-demi_length, 0))

        for scene, seg_scene, depth_scene in zip(self.scenes, self.seg_scenes, self.depth_scenes):
            Pn = read_calib_file(scene / 'calib.txt')
            intrinsics = np.reshape(Pn['P2'].astype(np.float32), (3, 4))[:, :3]

            images = sorted(scene.files('*.png'))
            segmentations = sorted(seg_scene.files('*.png'))
            depths = sorted(depth_scene.files('*.png'))
            transformations_paths = sorted(seg_scene.files('*.txt'))

            for i in range(demi_length, len(transformations_paths)-demi_length):
                sample = {
                    'intrinsics': intrinsics,
                    'tgt': images[i + offset],
                    'ref_imgs':[],
                    'seg':segmentations[i + offset],
                    'ref_seg': [],
                    'depth': depths[i + offset],
                    'ref_depths': [],
                    'transformations': []
                }

                for j in shifts:
                    sample['ref_depths'].append(depths[i + j])
                    sample['ref_imgs'].append(images[i+j])
                    sample['ref_seg'].append(segmentations[i+j])

                    transformation = read_transformations(transformations_paths[i+j])
                    sample['transformations'].append(transformation)

                sequence_set.append(sample)

            if is_train:
                random.shuffle(sequence_set)

            self.samples = sequence_set

    def crawl_folders(self, sequence_length, is_train=True):
        print("crawling...")
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)

        shifts = list(range(-demi_length, 0))

        for scene, seg_scene in zip(self.scenes, self.seg_scenes):
            # intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))

            Pn = read_calib_file(scene / 'calib.txt')

            intrinsics = np.reshape(Pn['P2'].astype(np.float32), (3, 4))[:, :3]

            imgs = sorted(scene.files('*.png'))
            segmentations = sorted(seg_scene.files('*.png'))

            # boundaries = sorted(boundary_scene.files('*.png'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                          'seg': segmentations[i], 'ref_seg': [], 'ref_boundaries': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                    sample['ref_seg'].append(segmentations[i + j])

                    # sample['ref_boundaries'].append(boundaries[i + j])
                sequence_set.append(sample)

        if is_train:
            random.shuffle(sequence_set)
        self.samples = sequence_set

    def crawl_folders_pix2pix(self):
        sequence_set = []
        for scene, seg_scene, depth_scene in zip(self.scenes, self.seg_scenes, self.depth_scenes):
            imgs = sorted(scene.files('*.png'))
            segmentations = sorted(seg_scene.files('*.png'))
            depths = sorted(depth_scene.files('*.png'))
            print(len(imgs), len(segmentations))
            for i in range(len(segmentations) - 1):
                sample = {'ref': imgs[i], 'seg': segmentations[i + 1], 'tgt': imgs[i + 1], 'depth': depths[i + 1]}
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        ref_imgs = []
        tgt_depth = 0
        ref_depths = []
        ref_img = 0
        # boundary = 0
        ref_segs = []
        # ref_boundaries = []
        intrinsics = 0
        A = 0
        B = 0
        intrinsics_inv = 0
        transformation = []

        if self.depth_pred:
            tgt_img = load_as_float(sample['tgt'])
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            ref_segs = [load_as_float(ref_s) for ref_s in sample['ref_seg']]
            tgt_seg = load_as_float(sample['seg'])
            tgt_depth = load_as_float(sample['depth'])
            ref_depths = [load_as_float(ref_d) for ref_d in sample['ref_depths']]

            if self.transform is not None:
                imgs, _ = self.transform([tgt_img] + ref_imgs, None)
                segs, _ = self.transform([tgt_seg] + ref_segs, None)

                depths, _ = self.transform([tgt_depth] + ref_depths, None)

                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
                tgt_seg = segs[0]
                ref_segs = segs[1:]
                tgt_depth = depths[0]
                ref_depths = depths[1:]

            transformation = np.copy(sample['transformations'])
            transformation = torch.from_numpy(transformation).float()
            intrinsics = np.copy(sample['intrinsics'])
            intrinsics_inv = np.linalg.inv(intrinsics)

        elif not self.pix2pix:
            tgt_img = load_as_float(sample['tgt'])
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            ref_segs = [load_as_float(ref_s) for ref_s in sample['ref_seg']]
            seg = load_as_float(sample['seg'])            # boundary = load_as_float(sample['boundary'])

            if self.transform is not None:
                imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
                segs, intrinsics = self.transform([seg] + ref_segs, np.copy(sample['intrinsics']))
                # boundaries, intrinsics = self.transform([boundary] + ref_boundaries, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
                tgt_seg = segs[0]
                ref_segs = segs[1:]
            else:
                intrinsics = np.copy(sample['intrinsics'])

            intrinsics_inv = np.linalg.inv(intrinsics)

        else:
            tgt_img = load_as_float(sample['tgt'])
            ref_img = load_as_float(sample['ref'])
            tgt_seg = load_as_float(sample['seg'])
            depth = load_as_float(sample['depth'])

            depth = torch.from_numpy(depth).float() / 255
            depth.unsqueeze_(0)

            # print(depth.size())
            # depth = depth.expand(1, w, h)

            # print("tgt shape:", tgt_img.shape)
            # print("ref shape:", ref_img.shape)

            if self.transform is not None:
                tgt_img = self.transform(tgt_img)
                ref_img = self.transform(ref_img)

                seg = self.transform(tgt_seg)
                # print(seg.size())
                # depth = self.transform(depth)

                # print("----", depth.size())

            if self.p2p_benchmark:
                A = ref_img
                B = tgt_img
            else:
                A = torch.cat((ref_img, tgt_seg), 0)

                A = torch.cat((A, depth), 0)
                B = tgt_img

        return tgt_img, ref_imgs, tgt_seg, ref_segs, \
               intrinsics, intrinsics_inv, sample, \
               {'A': A, 'B': B}, transformation, tgt_depth, ref_depths

    # def __getitem__(self, index):
    #     sample = self.samples[index]
    #     imgs = [load_as_float(img) for img in sample['imgs']]
    #     if self.transform is not None:
    #         imgs, intrinsics, poses = self.transform([imgs], np.copy(sample['intrinsics']), np.copy(sample['poses']))
    #     else:
    #         intrinsics = np.copy(sample['intrinsics'])
    #         poses = np.copy(sample['poses'])
    #     return imgs, intrinsics, np.linalg.inv(intrinsics), poses

    def __len__(self):
        return len(self.samples)
