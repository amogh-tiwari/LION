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
from vae_infer_utils.run_utils import run_on_dataset, run_on_ho3d_subset
from vae_infer_utils.data_utils import gather_evaluation_assets

@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def infer(args, config):
    args, config, trainer, writer, nparam = setup(args, config)
    trainer.model.eval()
    batch_size=2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir_base = '/scratch/clear/atiwari/datasets/ho3d_v3_processing/HO3D_v3'
    
    # run_on_ho3d_subset(trainer, batch_size, device, out_dir_base=out_dir_base)
    # run_on_dataset(trainer, 'ho3d_2048')
    # Scaling params; Mean: [[[-0.00059158 -0.00091937 -0.00367902]]] | Std: [[[0.03672426]]]
    
    in_dir_grab = '../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/grab_2048/'
    in_fps_grab = gather_evaluation_assets(in_dir_grab)
    out_fps_grab = [fp.replace("transformed_assets", "lion_embeds") for fp in in_fps_grab]
    
    grab_normalization_params = {
        # 'all_points_mean': torch.tensor([[[-0.0012,  0.0006, -0.0026]]]).to(device='cuda:0', dtype=torch.float32),
        'all_points_mean': np.asarray([[[-0.0012,  0.0006, -0.0026]]]),
        # 'all_points_std': torch.tensor([[[0.0338]]]).to(device='cuda:0', dtype=torch.float32)
        'all_points_std': np.asarray([[[0.0338]]])
    }

    in_dir_ho3d = '../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/ho3d_2048/'
    in_fps_ho3d = gather_evaluation_assets(in_dir_ho3d)
    out_fps_ho3d = [fp.replace("transformed_assets", "lion_embeds") for fp in in_fps_ho3d]
    breakpoint()
    ho3d_normalization_params = {
    # 'all_points_mean': torch.tensor([[[-0.00059158 -0.00091937 -0.00367902]]]).to(device=device, dtype=torch.float32),
    'all_points_mean': np.asarray([[[-0.00059158 -0.00091937 -0.00367902]]]),
    # 'all_points_std': torch.tensor([[[0.03672426]]]).to(device=device, dtype=torch.float32),
    'all_points_std': np.asarray([[[0.03672426]]]),
    }

    run_on_dataset(trainer, 
                   'custom', 
                   in_fp=in_fps_ho3d,  # ATTENTION: Don't forget to change BELOW params
                   out_fp=out_fps_ho3d, # ATTENTION: Don't forget to change above params
                   normalization_params=ho3d_normalization_params # ATTENTION: Don't forget to change above params
                   )
    
    # run_on_dataset(trainer, 
    #                'custom', 
    #                in_fp='/scratch/clear/atiwari/datasets/ho3d_v3_processing/models_sampled/verts_2048/002_master_chef_can/textured_simple.obj',
    #                out_fp='./temp_outputs/002_master_chef_can/'
    #                )
    

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
