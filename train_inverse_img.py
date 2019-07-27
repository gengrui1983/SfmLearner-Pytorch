import argparse
import csv
import path
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from path import Path
from tensorboardX import SummaryWriter

from utils import tensor2array, save_checkpoint, save_path_formatter

parser = argparse.ArgumentParser(description='Invert training from depth to image',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--seg', metavar='DIR', help='path to segs')
parser.add_argument('--transformation', metavar='DIR', help='path to dir')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='N', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on valuation set')
parser.add_argument('--train_mode', dest='train_mode', action='store_true',
                    help='force check training model')
parser.add_argument('--output-dir', dest='output_dir', default=None, metavar='PATH', help='path to output')
parser.add_argument('--pretrained-model',dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pretrained model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-output', action='store-true', help='will log outputs and predicted image at validation step')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting', metavar='N', default=0)

best_error = -1
n_iter = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

if __name__ == '__main__':
    main()