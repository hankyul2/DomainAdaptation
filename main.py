import datetime
import sys
import os
import argparse

import numpy as np
import random

import torch

from src.log import get_log_name


parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='Enter which gpu you want to use')
parser.add_argument('-r', '--random_seed', type=int, default=None, help='Enter random seed')
parser.add_argument('-m', '--model_name', type=str.upper, default='', choices=[
    'BASE',
    'DANN',
    'CDAN', 'CDAN_E',
    'DANN_BSP', 'CDAN_BSP', 'CDAN_E_BSP',
    'FIXBI',
], help='Enter model name')
parser.add_argument('-s', '--src', type=str, default='', help='Enter source dataset')
parser.add_argument('-t', '--tgt', type=str, default='', help='Enter target dataset')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Enter batch size for train step')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Enter the number of workers per dataloader')
parser.add_argument('-l', '--lr', type=float, default=1e-2, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=int, default=100, help='Enter the number of epoch')
parser.add_argument('-i', '--iter', type=int, default=1, help='Enter the number of iteration you want to run')
parser.add_argument('--save_best_result', action='store_true', help='If specify, it will save best result into readme')


def init_seed(random_seed):
    if args.random_seed is None:
        return
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def init_cli_arg(args):
    args.log_name = get_log_name(args)
    args.start_time = datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
    args.gpu_id_list = list(map(int, args.gpu_id.split(',')))
    args.world_size = max(1, len(args.gpu_id_list))
    args.rank = 0
    args.total_batch_size = args.batch_size * args.world_size
    args.is_multi_gpu = args.world_size > 1


def show_info(args):
    print('DEVICE is {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))
    print('is multi gpu? {}'.format(args.is_multi_gpu))
    print('total batch size is {}'.format(args.total_batch_size))
    print('world size is {}'.format(args.world_size))
    print('Model name is {}'.format(args.model_name))
    print('Dataset name is {}, {}'.format(args.src, args.tgt))
    print('start time is {}'.format(args.start_time))


def init(args):
    sys.path.append('.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    init_seed(args.random_seed)
    init_cli_arg(args)
    show_info(args)


if __name__ == '__main__':
    args = parser.parse_args()
    init(args)

    if args.save_best_result:
        from src.log import run

    elif args.model_name in ['BASE']:
        from src.train import run

    elif args.model_name in ['DANN']:
        from src.train_da.train_dann import run

    elif args.model_name in ['CDAN', 'CDAN_E']:
        from src.train_da.train_cdan import run

    elif args.model_name in ['CDAN_BSP', 'CDAN_E_BSP', 'DANN_BSP', 'DANN_E_BSP']:
        from src.train_da.train_bsp import run

    elif args.model_name in ['FIXBI']:
        from src.train_da.train_fixbi import run

    for iter in range(args.iter):
        run(args)
        init_cli_arg(args)