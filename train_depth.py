import argparse
import csv
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import custom_transforms
import models
from logger import TermLogger, AverageMeter
from utils import tensor2array, save_checkpoint_depth_seg, save_path_formatter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument("--boundary", metavar='DIR', help="The directory of boundary images")
parser.add_argument("--segmentation", metavar='DIR', help="The directory of segmentation images")
parser.add_argument("--depth", metavar='DIR', help="The directory of depth images.")
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train_mode', dest='train_mode', action='store_true',
                    help='force check training mode')
parser.add_argument('--odemetry', dest='odemetry', action='store_true',
                    help='using odemetry dataset')
parser.add_argument('--output-dir', dest='output_dir', default=None, metavar='PATH', help='path of output image')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-depth-seg-net', dest='pretrained_depth_seg_net', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global  best_error, n_iter, device
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints/depth_seg'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epoch = 0

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    normalize = custom_transforms.Normalize(mean=[.5, .5, .5],
                                            std=[.5, .5, .5])
    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        args.segmentation,
        args.depth,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        odemetry=args.odemetry,
        depth_prediction=True
    )

    val_set = SequenceFolder(
        args.data,
        segmentation=args.segmentation,
        depth=args.depth,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        odemetry=args.odemetry,
        depth_prediction=True
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    print('=> creating model')

    depth_seg_net = models.DepthSegNet().to(device)

    if args.pretrained_depth_seg_net:
        print('=> using pre-trained weights for depth-seg-net')
        weights = torch.load(args.pretrained_depth_seg_net)
        depth_seg_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        depth_seg_net.init_weights()

    cudnn.benchmark = True
    depth_seg_net = torch.nn.DataParallel(depth_seg_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': depth_seg_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])

    logger = TermLogger(n_epochs=args.epochs,
                        train_size=min(len(train_loader),
                                       args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        logger.reset_train_bar()
        train_loss = train(args, train_loader, depth_seg_net, optimizer, args.epoch_size, logger, training_writer, output_writers)
        logger.train_writer.write(' * Avg {}'.format(train_loss))

        logger.reset_valid_bar()
        errors, error_names = validate(args, val_loader, depth_seg_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[2:9], errors[2:9]))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint_depth_seg(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': depth_seg_net.state_dict()
            }, is_best
        )

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, depth_seg_net, optimizer, epoch_size, logger, train_writer, output_writers):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    depth_seg_net.train()

    end = time.time()
    logger.train_bar.update(0)

    criterion = nn.L1Loss()

    for i, (tgt_img, ref_imgs, tgt_seg, ref_segs,
            intrinsics, intrinsics_inv, sample, object,
            transformation, tgt_depth, ref_depths) in enumerate(train_loader):
        data_time.update(time.time() - end)

        tgt_seg = tgt_seg.to(device)
        ref_depths = [image.to(device) for image in ref_depths]
        ref_segs = [image.to(device) for image in ref_segs]
        tgt_depth = tgt_depth.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        transformation = [t.to(device) for t in transformation]

        pred = depth_seg_net(ref_segs, ref_depths,
                             intrinsics, intrinsics_inv, transformation)
        l1 = criterion(pred, tgt_seg)

        loss = l1

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('error', l1.item(), n_iter)

        # print(args.training_output_freq, n_iter)
        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:
            train_writer.add_image('train Input seg', tensor2array(tgt_seg[0]), n_iter)
            train_writer.add_image('train Input depth', tensor2array(tgt_depth[0], max_value=1), n_iter)

            for k, pred_seg in enumerate(pred):
                train_writer.add_image('train output normalized {}'.format(k),
                                       tensor2array(pred[k]),
                                       n_iter)

        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate(args, val_loader, depth_seg_net, epoch, logger, output_writers=[]):
    global n_iter, device
    batch_time = AverageMeter()
    losses = AverageMeter(precision=4)
    depth_seg_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    criterion = nn.L1Loss()

    for i, (tgt_img, ref_imgs, tgt_seg, ref_segs,
            intrinsics, intrinsics_inv, sample, object,
            transformation, tgt_depth, ref_depths) in enumerate(val_loader):

        tgt_seg = tgt_seg.to(device)
        ref_depths = [image.to(device) for image in ref_depths]
        ref_segs = [image.to(device) for image in ref_segs]
        # tgt_depth = tgt_depth.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        transformation = [t.to(device) for t in transformation]

        pred = depth_seg_net(ref_segs, ref_depths,
                             intrinsics, intrinsics_inv, transformation)
        l1 = criterion(pred, tgt_seg)

        loss = l1.item()

        if epoch == 0 and i < len(output_writers):
            for j, ref in enumerate(ref_segs):
                output_writers[i].add_image('val Input seg {}'.format(j), tensor2array(ref_segs[0][0]), 0)
                output_writers[i].add_image('val Input depth {}'.format(j), tensor2array(ref_depths[0][0], max_value=1), 1)

            output_writers[i].add_image('val DepthSegNet Normalized',
                                        tensor2array(pred[0]), epoch)

        if i < len(val_loader) - 1:
            for j, ref in enumerate(ref_segs):
                # print(i, j, pred[0].shape, pred[0, j].size(0))
                if i < len(output_writers):
                    output_writers[i].add_image('val Pred Output {}'.format(j), tensor2array(pred[0]), epoch)

        losses.update([loss])

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))

    return losses.avg, ['L1 Loss']


if __name__ == '__main__':
    main()
