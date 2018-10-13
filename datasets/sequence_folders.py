import os

import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


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

    def __init__(self, root, seed=None, train=True, sequence_length=2, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        print(os.path.abspath("."), self.root)
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.samples = None
        # self.crawl_folders(sequence_length)
        self.crawl_folders(sequence_length)

    # def crawl_folders(self, sequence_length):
    #     sequence_set = []
    #     demi_length = (sequence_length-1)//2
    #     shifts = list(range(-demi_length, demi_length + 1))
    #     shifts.pop(demi_length)
    #     for scene in self.scenes:
    #         intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
    #         imgs = sorted(scene.files('*.jpg'))
    #         if len(imgs) < sequence_length:
    #             continue
    #         for i in range(demi_length, len(imgs)-demi_length):
    #             sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
    #             for j in shifts:
    #                 sample['ref_imgs'].append(imgs[i+j])
    #             sequence_set.append(sample)
    #     random.shuffle(sequence_set)
    #     self.samples = sequence_set

    def crawl_folders(self, sequence_length):
        sequence_set = []
        for scence in self.scenes:
            intrinsics = np.genfromtxt(scence/'cam.txt').astype(np.float32).reshape((3, 3))
            poses = np.genfromtxt(scence/'poses.txt').astype(np.float32)
            imgs = sorted(scence.files('*.jpg'))
            for i in range(len(imgs) - 1):
                sample = {'intrinsics': intrinsics, 'imgs': [], 'poses': []}
                for j in range(sequence_length):
                    sample['imgs'].append(imgs[i+j])
                    sample['poses'].append(poses[i+j].reshape((3, 4)))
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    # def __getitem__(self, index):
    #     sample = self.samples[index]
    #     tgt_img = load_as_float(sample['tgt'])
    #     ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
    #     if self.transform is not None:
    #         imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
    #         tgt_img = imgs[0]
    #         ref_imgs = imgs[1:]
    #     else:
    #         intrinsics = np.copy(sample['intrinsics'])
    #     return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['imgs']]
        if self.transform is not None:
            imgs, intrinsics, poses = self.transform([imgs], np.copy(sample['intrinsics']), np.copy(sample['poses']))
        else:
            intrinsics = np.copy(sample['intrinsics'])
            poses = np.copy(sample['poses'])
        return imgs, intrinsics, np.linalg.inv(intrinsics), poses

    def __len__(self):
        return len(self.samples)
