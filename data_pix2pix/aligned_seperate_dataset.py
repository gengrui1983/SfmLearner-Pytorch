import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class AlignedSeperateDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert (opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
            w_offset: w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
            w + w_offset:w + w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5))
