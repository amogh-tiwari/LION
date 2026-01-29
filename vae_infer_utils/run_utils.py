import torch

import os
import numpy as np
import trimesh

from vae_infer_utils.data_utils import gather_data
from vae_infer_utils.normalization import normalize


def run_on_dataset(trainer, data_name, in_fp=None, out_fp=None):
    batch_size = 1
    all_pts, all_out_fps = gather_data(data_name, in_fp, out_fp)

    print(f"#### ALL PTS INFO ### | Shape: {all_pts.shape} | Mean: {all_pts.mean()} | Max: {all_pts.max()} | Min: {all_pts.min()} | Std: {all_pts.std()}")
    all_pts_norm, all_pts_mean, all_pts_std = normalize(all_pts)
    print(f"#### ALL PTS INFO ### | Shape: {all_pts_norm.shape} | Mean: {all_pts_norm.mean()} | Max: {all_pts_norm.max()} | Min: {all_pts_norm.min()} | Std: {all_pts_norm.std()}")
    

    for i in range(0, int(np.ceil(len(all_pts_norm) / batch_size))):
        start_idx=i*batch_size
        end_idx=min( (i+1) * batch_size, len(all_pts_norm) )
        curr_pts = all_pts[start_idx:end_idx]
        curr_pts_norm = all_pts_norm[start_idx:end_idx]
        curr_out_fps = all_out_fps[start_idx:end_idx]

        inp = torch.tensor(curr_pts_norm).to(device='cuda', dtype=torch.float32)
        out = trainer.model.recont(inp)

        for j in range(len(curr_pts_norm)):
            os.makedirs(curr_out_fps[j], exist_ok=True)
            trimesh.PointCloud(curr_pts[j]).export(os.path.join(curr_out_fps[j], curr_out_fps[j].rstrip('/').split("/")[-1]+"_input.obj"))
            trimesh.PointCloud(curr_pts_norm[j]).export(os.path.join(curr_out_fps[j], curr_out_fps[j].rstrip('/').split("/")[-1]+"_input_norm.obj"))
            trimesh.PointCloud(out['x_0_pred'][j].detach().cpu().numpy()).export(os.path.join(curr_out_fps[j], curr_out_fps[j].rstrip('/').split("/")[-1]+"_out.obj"))

            pred_info = {
                    'mu': out['latent_list'][0][1][j].detach().cpu().numpy(),
                    'sigma': out['latent_list'][0][2][j].detach().cpu().numpy(),
                }
            np.savez_compressed(os.path.join(curr_out_fps[j], curr_out_fps[j].rstrip('/').split("/")[-1]+"_pred_info.npz"), **pred_info)
