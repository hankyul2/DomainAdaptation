import sys
import os
import argparse

import numpy as np
import random

import torch

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
parser.add_argument('-l', '--lr', type=float, default=3e-3, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=int, default=50, help='Enter the number of epoch')
parser.add_argument('-c', '--ncrop', type=int, default=10, help='Enter the number of crop for final evaluation')
parser.add_argument('-i', '--iter', type=int, default=1, help='Enter the number of iteration you want to run')
parser.add_argument('--use_ncrop_for_valid', action='store_true',
                    help='If specify, it will use ncrop for validation')
parser.add_argument('--save_best_result', action='store_true', help='If specify, it will save best result into readme')


def init(args):
    sys.path.append('.')
    if args.random_seed:
        fix_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))


def fix_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

if __name__ == '__main__':
    args = parser.parse_args()
    init(args)
    print('Model name is {}'.format(args.model_name))

    if args.save_best_result:
        from src.log import run

    elif args.model_name in ['BASE']:
        from src.train import run

    elif args.model_name in ['DANN']:
        from src.train_dann import run

    elif args.model_name in ['CDAN', 'CDAN_E']:
        from src.train_cdan import run

    elif args.model_name in ['CDAN_BSP', 'CDAN_E_BSP', 'DANN_BSP', 'DANN_E_BSP']:
        from src.train_bsp import run

    elif args.model_name in ['FIXBI']:
        from src.train_fixbi import run

    for iter in range(args.iter):
        run(args)