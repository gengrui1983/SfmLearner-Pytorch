import os
import random

import numpy as np
import torch
import torch.utils.data as data
from imageio import imread
from path import Path


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

    def __init__(self, root, segmentation, boundary=None, seed=None, train=True, sequence_length=2, transform=None,
                 target_transform=None, pix2pix=False):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.segmentation = Path(segmentation)
        scene_list_path = None
        if not pix2pix:
            # scene_list_path = self.root / 'train.txt' if train else self.root / 'train_pix2pix.txt'
            scene_list_path = self.root / 'train.txt' if train else self.root / 'train_pix2pix_val.txt'
        else:
            scene_list_path = self.root / 'train_pix2pix.txt' if train else self.root / 'val.txt'
        # if boundary:
        #     self.boundary = Path(boundary)
        #     self.boundary_scenes = [self.boundary / folder[:-1] for folder in open(scene_list_path)]

        print(os.path.abspath("."), self.root)
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.seg_scenes = [self.segmentation / folder[:-1] for folder in open(scene_list_path)]

        self.transform = transform
        self.samples = None
        self.pix2pix = pix2pix
        # self.crawl_folders(sequence_length)
        if pix2pix:
            self.crawl_folders_pix2pix()
        else:
            self.crawl_folders(sequence_length, train)

    def crawl_folders(self, sequence_length, is_train=True):
        print("crawling...")
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene, seg_scene in zip(self.scenes, self.seg_scenes):
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
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
        # print(self.samples)
        print("length:", len(sequence_set))

    def crawl_folders_pix2pix(self):
        sequence_set = []
        for scene, seg_scene in zip(self.scenes, self.seg_scenes):
            imgs = sorted(scene.files('*.jpg'))
            segmentations = sorted(seg_scene.files('*.png'))
            print(len(imgs), len(segmentations))
            for i in range(len(segmentations) - 1):
                sample = {'ref': imgs[i], 'seg': segmentations[i], 'tgt': imgs[i + 1]}
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        ref_imgs = []
        ref_img = 0
        # boundary = 0
        ref_segs = []
        # ref_boundaries = []
        intrinsics = 0
        A = 0
        B = 0
        intrinsics_inv = 0

        if not self.pix2pix:
            tgt_img = load_as_float(sample['tgt'])
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            ref_segs = [load_as_float(ref_s) for ref_s in sample['ref_seg']]
            # ref_boundaries = [load_as_float(ref_b) for ref_b in sample['ref_boundaries']]
            seg = load_as_float(sample['seg'])
            # boundary = load_as_float(sample['boundary'])

            if self.transform is not None:
                imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
                segs, intrinsics = self.transform([seg] + ref_segs, np.copy(sample['intrinsics']))
                # boundaries, intrinsics = self.transform([boundary] + ref_boundaries, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
                seg = segs[0]
                ref_segs = segs[1:]
            else:
                intrinsics = np.copy(sample['intrinsics'])

            intrinsics_inv = np.linalg.inv(intrinsics)

        else:
            tgt_img = load_as_float(sample['tgt'])
            ref_img = load_as_float(sample['ref'])
            seg = load_as_float(sample['seg'])

            # print("tgt shape:", tgt_img.shape)
            # print("ref shape:", ref_img.shape)

            if self.transform is not None:
                tgt_img = self.transform(tgt_img)
                ref_img = self.transform(ref_img)
                seg = self.transform(seg)

            A = torch.cat((ref_img, seg), 0)
            B = tgt_img

        # print("index", index, "seg", seg.shape, "tgt_img", tgt_img.shape)
        # return tgt_img, ref_imgs, seg, intrinsics, np.linalg.inv(intrinsics)
        return tgt_img, ref_imgs, seg, ref_segs, \
               intrinsics, intrinsics_inv, sample, \
               {'A': A, 'B': B}

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
