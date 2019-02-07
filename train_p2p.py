import time

import torch
import torch.optim
import torch.utils.data
from path import Path
from tensorboardX import SummaryWriter

import custom_transforms
from models_pix2pix.models import create_model
from pix2pix_options.train_options import TrainOptions
from util.visualizer import Visualizer

# parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument("--segmentation", metavar='DIR', help="The directory of segmentation images")
# parser.add_argument("--boundary", metavar='DIR', help="The directory of boundary images")
# parser.add_argument('--dataset-format', default='sequential', metavar='STR',
#                     help='dataset format, stacked: stacked frames (from original TensorFlow code) \
#                     sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
# parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
# parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
#                     help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
# parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
#                     help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
#                          ' zeros will null gradients outside target image.'
#                          ' border will only null gradients of the coordinate outside (x or y)')
# parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
#                     You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers')
# parser.add_argument('--epochs', default=200, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
#                     help='manual epoch size (will match dataset size if not set)')
# parser.add_argument('-b', '--batch-size', default=4, type=int,
#                     metavar='N', help='mini-batch size')
# parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum for sgd, alpha parameter for adam')
# parser.add_argument('--beta', default=0.999, type=float, metavar='M',
#                     help='beta parameters for adam')
# parser.add_argument('--weight-decay', '--wd', default=0, type=float,
#                     metavar='W', help='weight decay')
# parser.add_argument('--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--output-dir', dest='output_dir', default=None, metavar='PATH', help='path of output image')
# parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
#                     help='path to pre-trained dispnet model')
# parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
#                     help='path to pre-trained Exp Pose net model')
# parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
# parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
#                     help='csv where to save per-epoch train and valid stats')
# parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
#                     help='csv where to save per-gradient descent train stats')
# parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
# parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
# parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
# parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
# parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
#                     metavar='N', default=0)

opt = TrainOptions().parse()

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    # args = parser.parse_args()
    if opt.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif opt.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    # save_path = save_path_formatter(args, parser)
    save_path = Path(opt.save_path)
    opt.save_path = 'checkpoints/pix2pix' / save_path
    print('=> will save everything to {}'.format(opt.save_path))
    opt.save_path.makedirs_p()
    torch.manual_seed(opt.seed)
    # if args.evaluate:
    #     args.epochs = 0

    training_writer = SummaryWriter(opt.save_path)
    output_writers = []
    if opt.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(opt.save_path / 'valid' / str(i)))

    # Data loading code
    normalize = custom_transforms.NormalizeWithoutInstrinsics(mean=[0.5, 0.5, 0.5],
                                                              std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.ComposeWithoutINstrinsics([
        custom_transforms.ArrayToTensorWithoutInstrinsic(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(opt.data))
    train_set = SequenceFolder(
        opt.data,
        opt.segmentation,
        depth=opt.depth,
        transform=train_transform,
        seed=opt.seed,
        train=True,
        pix2pix=True
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if opt.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            opt.data,
            transform=valid_transform,
        )
    else:
        val_set = SequenceFolder(
            opt.data,
            segmentation=opt.segmentation,
            # boundary=opt.boundary,
            transform=valid_transform,
            seed=opt.seed,
            train=False,
            sequence_length=opt.sequence_length,
            pix2pix=True
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    dataset_size = len(train_loader)

    if opt.epoch_size == 0:
        opt.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, (_, _, _, _, _, _, _, data) in enumerate(train_loader):

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    main()
