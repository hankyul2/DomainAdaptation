import sys
import os
import argparse
from itertools import permutations
from pathlib import Path

import pandas as pd

import numpy as np
import random

import torch

parser = argparse.ArgumentParser(description='Knowledge Disillation')
parser.add_argument('-g', '--gpu_id', type=str, default='', help='Enter which gpu you want to use')
parser.add_argument('-r', '--random_seed', type=int, default=3, help='Enter random seed')
parser.add_argument('-m', '--model_name', type=str.upper, default='', choices=[
    'DANN',
    'CDAN', 'CDAN_E',
    'DANN_BSP', 'CDAN_BSP', 'CDAN_E_BSP'
], help='Enter model name')
parser.add_argument('-s', '--src', type=str, default='', help='Enter source dataset')
parser.add_argument('-t', '--tgt', type=str, default='', help='Enter target dataset')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Enter batch size for train step')
parser.add_argument('-w', '--num_workers', type=int, default=4, help='Enter the number of workers per dataloader')
parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Enter learning rate')
parser.add_argument('-e', '--nepoch', type=float, default=50, help='Enter the number of epoch')
parser.add_argument('-c', '--ncrop', type=int, default=10, help='Enter the number of crop for final evaluation')
parser.add_argument('-B', '--benchmark', type=str, default=None,
                    choices=['all', 'office_31', 'office_home', 'digits'], help='Enter the benchmark name you want to test')
parser.add_argument('--use_ncrop_for_valid', action='store_true',
                    help='If specify, it will use ncrop for validation')


def init(args):
    sys.path.append('.')
    fix_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('DEVICE: {}'.format('cpu' if args.gpu_id == '' else args.gpu_id))


def fix_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


def benchmark(args, run, benchmark_result='benchmark_result'):
    Path(benchmark_result).mkdir(exist_ok=True)
    benchmark = {
        'office_31':['amazon', 'dslr', 'webcam'],
        'office_home':['art', 'clip_art', 'product', 'real_world'],
        'digit':['mnist', 'svhn', 'usps']
    }

    if args.benchmark == 'all':
        pass
    else:
        result = {}
        for iter, (src, tgt) in enumerate(permutations(benchmark[args.benchmark], 2)):
            fix_seed(iter+1)
            args.src = src
            args.tgt = tgt
            best_acc = run(args)
            result['{}->{}'.format(src, tgt)] = [best_acc]
    df = pd.DataFrame(result)
    df.to_csv("{}/{}_{}.csv".format(benchmark_result, args.model_name, args.benchmark), index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    init(args)

    if args.model_name in ['DANN']:
        from src.train_dann import run
        print('Model name is {}'.format(args.model_name))

    elif args.model_name in ['CDAN', 'CDAN_E']:
        from src.train_cdan import run
        print('Model name is {}'.format(args.model_name))

    elif args.model_name in ['CDAN_BSP', 'CDAN_E_BSP', 'DANN_BSP', 'DANN_E_BSP']:
        from src.train_bsp import run
        print('Model name is {}'.format(args.model_name))

    if args.benchmark:
        benchmark(args, run)
    else:
        run(args)