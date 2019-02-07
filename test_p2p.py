import os

import numpy as np
import torch.utils.data

import custom_transforms
# from options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
from datasets.sequence_folders import SequenceFolder
from models_pix2pix.models import create_model
from pix2pix_options.train_options import TrainOptions
from util import html
from util.ssim import SSIM
from util.visualizer import Visualizer

train = TrainOptions()
train.isTrain = False
opt = train.parse()
opt.isTrain = False

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# data_loader = CreateDataLoader(opt)


normalize = custom_transforms.NormalizeWithoutInstrinsics(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])

train_transform = custom_transforms.ComposeWithoutINstrinsics([
    custom_transforms.ArrayToTensorWithoutInstrinsic(),
    normalize
])

test_set = SequenceFolder(
    opt.data,
    opt.segmentation,
    opt.depth,
    transform=train_transform,
    seed=opt.seed,
    train=False,
    pix2pix=True
)

dataset_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False,
    num_workers=opt.workers, pin_memory=True)

# dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

results = []
for i, (tgt_img, _, _, _, _, _, sample, data) in enumerate(dataset_loader):
    if i >= opt.how_many != -1:
        break

    model.set_input(data)

    model.test()
    visuals = model.get_current_visuals()
    # img_path = model.get_image_paths()

    tgt_path = sample['tgt'][0].split('/')[-2:]
    file_path = tgt_path[-1].split('.')[0]
    scence_path = tgt_path[-2]

    img_path = scence_path + "/" + file_path
    # img_path = "/".join(tgt_path)

    # print(Path(img_path))

    real_A = visuals['real_A']
    real_B = visuals['real_B']
    fake_B = visuals['fake_B']

    real_A = np.einsum('WHC->CWH', real_A)
    real_B = np.einsum('WHC->CWH', real_B)
    fake_B = np.einsum('WHC->CWH', fake_B)

    real_B = torch.from_numpy(real_B).cuda()
    fake_B = torch.from_numpy(fake_B).cuda()

    real_B = real_B.unsqueeze(0)
    fake_B = fake_B.unsqueeze(0)

    real_B = real_B.type('torch.DoubleTensor')
    fake_B = fake_B.type('torch.DoubleTensor')

    ssim = SSIM()
    r = ssim(real_B, fake_B)

    if r < 1:
        results.append(r)
        print("****************", i, r, sum(results) / i)

    # print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()

