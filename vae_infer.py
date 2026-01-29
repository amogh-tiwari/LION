import importlib
import argparse
from loguru import logger
from comet_ml import Experiment
import torch
import numpy as np
import os
import sys
import torch.distributed as dist
from torch.multiprocessing import Process
from default_config import cfg as config
from utils import exp_helper, io_helper
from utils import utils

from train_dist import get_args
from easydict import EasyDict as edict

import trimesh

from tqdm import tqdm

from vae_infer_utils.setup import setup
from vae_infer_utils.run_utils import run_on_dataset

@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def infer(args, config):
    args, config, trainer, writer, nparam = setup(args, config)
    trainer.model.eval()

    # run_on_dataset(trainer, 'ho3d_2048')
    # Scaling params; Mean: [[[-0.00059158 -0.00091937 -0.00367902]]] | Std: [[[0.03672426]]]

    run_on_dataset(trainer, 
                   'custom', 
                   in_fp='/scratch/clear/atiwari/datasets/ho3d_v3_processing/models_sampled/verts_2048/021_bleach_cleanser/textured_simple.obj',
                   out_fp='./temp_outputs/021_bleach_cleanser/'
                   )
    

# Copied get_args() and __main__ block from train_dist.py, keeping it unchanged
if __name__ == '__main__':
    args, config = get_args()
    args.ntest = int(args.ntest) if args.ntest is not None else None
    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            logger.info('In Rank={}', rank)
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_size = global_size
            args.global_rank = global_rank
            logger.info('Node rank %d, local proc %d, global proc %d' %
                        (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes,
                        args=(global_rank, global_size, main, args, config))
            p.start()
            processes.append(p)

        for p in processes:
            logger.info('join {}', args.local_rank)
            p.join()
    else:
        # for debugging
        args.distributed = False
        args.global_size = 1
        utils.init_processes(0, size, infer, args, config)
    logger.info('should end now')
    # if args.distributed:
    #    logger.info('destroy_process_group')
    #    dist.destroy_process_group()
