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

NUM_VERTICES_TO_SAMPLE = 2048

def sample_vertices(verts, n_samples):
    N = verts.shape[0]
    if N > n_samples:
        sampling_indices = np.random.choice(verts.shape[0], size=n_samples, replace=False)
    else:
        sampling_indices = np.random.choice(verts.shape[0], size=n_samples, replace=True)

    sampled_verts = verts[sampling_indices]
    
    return sampled_verts

def normalize(all_points, normalization_type):
    input_dim = 3
    if normalization_type == 'normalize_shape_box':
        # Below code is used to get right additive (subratctive) and scaling factors to get inputs in range [-1,1]
        # Suppose, x \in [a,b]
        # on subtracting the mean (a+b)/2, now x \in [(a-b)/2, (b-a)/2]
        # now, on dividing by (b-a)/2, now x \in [-1, 1]

        B, N = all_points.shape[:2]
        all_points_mean = (  # B,1,3 # Simple mean by (min + max) / 2
            (torch.amax(all_points, axis=1)).reshape(B, 1, input_dim) +
            (torch.amin(all_points, axis=1)).reshape(B, 1, input_dim)) / 2
        all_points_std = torch.amax(  # B,1,1 # Not standard deviation, rather the scaling factor to be used later to get in range of (-1,1)
            ((torch.amax(all_points, axis=1)).reshape(B, 1, input_dim) -
                (torch.amin(all_points, axis=1)).reshape(B, 1, input_dim)),
            axis=-1).reshape(B, 1, 1) / 2
    else:
        print("!!! WARNING !!! Unknown normalization type, returning unnormalized points")
        # add some 'identity normalization'.
        exit()

    all_points = (all_points - all_points_mean) / all_points_std
    
    return all_points, all_points_mean, all_points_std

def fetch_vertices(inp_verts, normalization_type):
    if inp_verts.shape[0] != NUM_VERTICES_TO_SAMPLE: # inp_verts: Nx3
        sampled_verts = sample_vertices(inp_verts, NUM_VERTICES_TO_SAMPLE)
    else:
        sampled_verts = inp_verts

    sampled_verts = torch.tensor(sampled_verts, device='cuda', dtype=torch.float32)[None]
    sampled_verts, all_pts_mean, all_pts_std = normalize(sampled_verts, normalization_type)
    return sampled_verts, all_pts_mean, all_pts_std

def load_file_and_fetch_vertices(pcd_fp, normalization_type):
    inp_pcd = trimesh.load(pcd_fp)
    sampled_verts = fetch_vertices(inp_pcd.vertices, normalization_type)
    return sampled_verts

def run_on_grab_sample_data(trainer, base_dir, tgt_obj_name):
    inp_pcd = trimesh.load(os.path.join(base_dir, tgt_obj_name))
    inp_verts = inp_pcd.vertices
    sampled_verts, all_points_mean, all_points_std = fetch_vertices(inp_verts, 'normalize_shape_box')

    pcd_mins = inp_verts.min(axis=0)
    pcd_maxs = inp_verts.max(axis=0)  # (B, 3)
    pcd_size = pcd_maxs - pcd_mins      # (B, 3) - [x_range, y_range, z_range]

    out = trainer.model.recont(sampled_verts)

    _ = trimesh.PointCloud(sampled_verts[0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/grab_testing/{tgt_obj_name.split('.')[0]}_input.obj")
    _ = trimesh.PointCloud(out['x_0_pred'][0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/grab_testing/{tgt_obj_name.split('.')[0]}_recon.obj")
    _ = trimesh.PointCloud(out['vis/latent_pts'][0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/grab_testing/latent_pts/{tgt_obj_name.split('.')[0]}_latent_pts.obj")

    s_idx=0 ##### TODO: Change if using batch size > 1.
    out_sample = {
        'mu': out['latent_list'][0][1][s_idx].detach().cpu().numpy(),
        'sigma': out['latent_list'][0][2][s_idx].detach().cpu().numpy(),
        'xyz_extent': pcd_size, # No s_idx indexing here as long as it's a single batch operation.
        'shift_factor': all_points_mean[s_idx][0].detach().cpu().numpy(),
        'scale_factor': all_points_std[s_idx][0].detach().cpu().numpy() # NOTE: 'std' here is a misnomer. It's actually the scale info.
    }
    out_fp = f"./outputs/recon_outputs/grab_testing/embeds/{tgt_obj_name.split('.')[0]}_embeds.npz"
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    np.savez_compressed(out_fp, **out_sample)

def run_on_file_path(trainer, in_fp):
    sampled_verts = load_file_and_fetch_vertices(in_fp, 'normalize_shape_box')
    tgt_obj_name = in_fp.split("/")[-1].split(".")[0]

    out = trainer.model.recont(sampled_verts)
    _ = trimesh.PointCloud(sampled_verts[0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/shapenet_testing/{tgt_obj_name}_input.obj")
    _ = trimesh.PointCloud(out['x_0_pred'][0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/shapenet_testing/{tgt_obj_name}_recon.obj")
    _ = trimesh.PointCloud(out['vis/latent_pts'][0].detach().cpu().numpy()).export(f"./outputs/recon_outputs/shapenet_testing/{tgt_obj_name}_latent_pts.obj")

def run_on_grab_full_data(trainer):
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

    ds_train, ds_train_loader = get_grab_dataloader(grab_cfg.dataset_dir, 'train', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=False, drop_last=False)
    ds_test, ds_test_loader = get_grab_dataloader(grab_cfg.dataset_dir, 'test', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=False, drop_last=False)
    ds_val, ds_val_loader = get_grab_dataloader(grab_cfg.dataset_dir, 'val', grab_cfg.batch_size, grab_cfg.n_workers, shuffle=False, load_on_ram=grab_cfg.load_on_ram, return_names=True, return_addnl_data=False, drop_last=False)

    # # matching_result = np.char.find(ds_train_loader.frame_names, '/scratch/clear/atiwari/datasets/grabnet_extract/data/train/data/s6/scissors_use_2/')
    # # tgt_idxs = np.where(matching_result == 0)[0]
    # ds_val_loader.frame_names = ds_val_loader.frame_names[-128:]
    # ds_test_loader.frame_names = ds_test_loader.frame_names[-128:]

    split_list = ['test', 'val', 'train']
    trainer.model.eval()
    for split in split_list:
        ds = {"test": ds_test, "val": ds_val, "train": ds_train}[split] # Choose the right data loader for the split.

        n_batches = len(ds)
        for i, (batch, frame_names) in tqdm(enumerate(ds), total=len(ds), desc=f"Encoding dataset {split} split"):
            batch = {k: v.to(grab_cfg.device) for k, v in batch.items()}
            inp_verts = batch['verts_object']
            pcd_mins = inp_verts.min(dim=1)[0]  # (B, 3)
            pcd_maxs = inp_verts.max(dim=1)[0]  # (B, 3)
            pcd_size = pcd_maxs - pcd_mins      # (B, 3) - [x_range, y_range, z_range]

            inp_verts, all_points_mean, all_points_std = normalize(batch['verts_object'], 'normalize_shape_box')
            # Disable gradient computation for inference
            with torch.no_grad():
                out = trainer.model.recont(inp_verts)
                
            for s_idx in range(len(frame_names)):
                out_fp = frame_names[s_idx].replace("grabnet_extract/", "grabnet_processing/lion_embeds_only_mu_sigma_after_normalizing_with_scale_factor/")
                
                # Extract and convert sample from batch
                # 'latent_list' structure: 
                #   - Level 1: [global_latent, local_latent] (2 elements)
                #   - Level 2: [z, mu, sigma] (3 elements each)
                #   - Level 3: tensor with shape (B, dim_latent)
                out_sample = {
                    # 'x_0_pred': out['x_0_pred'][s_idx].detach().cpu().numpy(),
                    # 'latent_list': [
                    #     [l2_item[s_idx].detach().cpu().numpy() for l2_item in l1_item] 
                    #     for l1_item in out['latent_list']
                    # ]
                    'mu': out['latent_list'][0][1][s_idx].detach().cpu().numpy(),
                    'sigma': out['latent_list'][0][2][s_idx].detach().cpu().numpy(),
                    'shift_factor': all_points_mean[s_idx][0].detach().cpu().numpy(),
                    'scale_factor': all_points_std[s_idx][0].detach().cpu().numpy() # NOTE: 'std' here is a misnomer. It's actually scaling factor.
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
    
    # run_on_grab_full_data(trainer)
    
    base_dir = "/scratch/clear/atiwari/datasets/grabnet_extract/tools/object_meshes/contact_meshes"
    for tgt_obj_name in tqdm(os.listdir(base_dir)):
        run_on_grab_sample_data(trainer, base_dir, tgt_obj_name)

    # run_on_file_path(trainer, '/scratch/clear/atiwari/lion/data/ShapeNetCore.v2.PC15k_visualizations/02691156/test/fff513f407e00e85a9ced22d91ad7027.obj')

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
