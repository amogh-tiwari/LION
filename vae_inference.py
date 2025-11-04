# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

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

def get_data():
    n_samples = 2048
    pcd = trimesh.load('/home/atiwari/projects/object_manipulation/assets/sample_data/test_objs/binoculars.ply')
    verts = pcd.vertices
    sampling_indices = np.random.choice(verts.shape[0], size=n_samples, replace=False)
    sampled_verts = verts[sampling_indices]


    return torch.tensor(sampled_verts, device='cuda', dtype=torch.float32)[None]

# inference_vae_main.py
# Just copy train_dist.py and modify ONLY the main() function

def main_logic(trainer):
    # inp_pcd = get_data()
    # out = trainer.model.recont(inp_pcd)
    # _ = trimesh.PointCloud(out['x_0_pred'][0].detach().cpu().numpy()).export("./outputs/recon_outputs/testing/recon_binoculars.obj")
    # _ = trimesh.PointCloud(out['vis/latent_pts'][0].detach().cpu().numpy()).export("./outputs/recon_outputs/testing/latent_pts_binoculars.obj")

    import sys
    sys.path.append("../object_manipulation")

    from data.grab_loader import get_grab_dataloader
    grab_cfg = {
        'dataset_dir': '/scratch/clear/atiwari/datasets/grabnet_extract/data',
        'batch_size': 128,
        'n_workers': 0,
        'load_on_ram': False,
        'device': 'cuda'
    }
    grab_cfg = edict(grab_cfg)

    ds_train, _ = get_grab_dataloader(grab_cfg.dataset_dir, 'train', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=True)
    ds_test, _ = get_grab_dataloader(grab_cfg.dataset_dir, 'test', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=True)
    ds_val, _ = get_grab_dataloader(grab_cfg.dataset_dir, 'val', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=True)

    split_list = ['test']
    trainer.model.eval()
    for split in split_list:
        ds = {"test": ds_test, "val": ds_val, "train": ds_train}[split] # Choose the right data loader for the split.
        for i, (batch, frame_names) in tqdm(enumerate(ds), total=len(ds), desc=f"Encoding dataset {split} split"):
            batch = {k: v.to(grab_cfg.device) for k, v in batch.items()}
            inp_verts = batch['verts_object']
            # Disable gradient computation for inference
            with torch.no_grad():
                out = trainer.model.recont(inp_verts)
                
            for s_idx in range(grab_cfg['batch_size']):
                out_fp = frame_names[s_idx].replace("grabnet_extract/", "grabnet_processing/lion_embeds/")
                
                # Extract and convert sample from batch
                # 'latent_list' structure: 
                #   - Level 1: [global_latent, local_latent] (2 elements)
                #   - Level 2: [z, mu, sigma] (3 elements each)
                #   - Level 3: tensor with shape (B, dim_latent)
                out_sample = {
                    'x_0_pred': out['x_0_pred'][s_idx].detach().cpu().numpy(),
                    'latent_list': [
                        [l2_item[s_idx].detach().cpu().numpy() for l2_item in l1_item] 
                        for l1_item in out['latent_list']
                    ]
                }
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)
                np.savez_compressed(out_fp, **out_sample)


@logger.catch(onerror=lambda _: sys.exit(1), reraise=False)
def main(args, config):
    # -- trainer -- #
    logger.info('use trainer: {}', config.trainer.type)
    trainer_lib = importlib.import_module(config.trainer.type)
    Trainer = trainer_lib.Trainer

    if config.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info(
            '\n\n' + '!'*30 + '\nWARNING: ths set_detect_anomaly is turned on, it can slow down the training! \n' + '!'*30)

    # -- command init -- #
    comet_key = config.comet_key
    _, writer = utils.common_init(args.global_rank,
                                  config.trainer.seed, config.save_dir, comet_key)
    trainer = Trainer(config, args)
    nparam = utils.count_parameters_in_M(trainer.model)
    logger.info('param size = %fM ' % nparam)

    # -- Load checkpoint (this part stays from original) -- #
    if args.resume or args.eval_generation:
        if args.pretrained is not None:
            trainer.start_epoch = trainer.resume(
                args.pretrained, eval_generation=args.eval_generation)
        else:
            raise NotImplementedError
    elif args.pretrained is not None:
        logger.info('Loading pretrained weights from: {}', args.pretrained)
        trainer.resume(args.pretrained)

    # Set to eval mode
    trainer.model.eval()
    
    logger.info('='*80)
    logger.info('VAE MODEL LOADED SUCCESSFULLY!')
    logger.info('='*80)
    logger.info('Model parameters: {:.2f}M', nparam)
    
    # TODO: Add your reconstruction code here
    # DO NOT call trainer.train_epochs() or trainer.eval_nll() etc.
    main_logic(trainer)


# Keep get_args() and __main__ completely unchanged from train_dist.py
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
        utils.init_processes(0, size, main, args, config)
    logger.info('should end now')
    # if args.distributed:
    #    logger.info('destroy_process_group')
    #    dist.destroy_process_group()
