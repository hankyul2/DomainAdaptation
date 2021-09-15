import os
import torch.distributed as dist
from torch.multiprocessing import spawn

from src.distributed.dist_wrapper import apply_dist


def init_process(rank, main, args):
    args.rank = rank
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    apply_dist(args.rank, args.world_size, args.log_name, args.start_time)
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    main(args)


def run_multi_gpus(main, args):
    spawn(fn=init_process, args=(main, args), nprocs=args.world_size)