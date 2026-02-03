import torch
from torch.utils import data

import os
import numpy as np
import trimesh

from tqdm import tqdm

from vae_infer_utils.data_utils import gather_data
from vae_infer_utils.normalization import normalize, normalize_ho3d


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

def run_on_ho3d_subset(trainer, batch_size, device, out_dir_base):
    import sys
    sys.path.append("../object_manipulation")
    from data.ho3d_loader import Ho3dData

    ho3d_train_set = Ho3dData('train_subset')
    ho3d_train_loader = data.DataLoader(ho3d_train_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    for idx, batch in tqdm(enumerate(ho3d_train_loader), total=len(ho3d_train_loader), desc='iterating over ho3d'):
        inp_verts = batch['verts_object']
        inp_verts_orig = inp_verts.clone()
        print(f"Shape: {inp_verts.shape} | Mean: {inp_verts.mean()} | Max: {inp_verts.max()} | Min: {inp_verts.min()} | Std: {inp_verts.std()}")
        inp_verts, all_points_mean, all_points_std = normalize_ho3d(batch['verts_object'], 'normalize_global')
        print(f"Shape: {inp_verts.shape} | Mean: {inp_verts.mean()} | Max: {inp_verts.max()} | Min: {inp_verts.min()} | Std: {inp_verts.std()}")
        
        inp_verts = inp_verts.to(device=device, dtype=torch.float32)
        out = trainer.model.recont(inp_verts)
        
        n_samples = inp_verts.shape[0]
        for j in range(n_samples):
            seq_name, frame_name = batch['file_path'][j].split("/")
            out_dir = os.path.join(out_dir_base, ho3d_train_set.split, seq_name, "lion_embeds")
            out_dir_viz = out_dir.replace("lion_embeds", "lion_embeds_viz")
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(out_dir_viz, exist_ok=True)
            pred_info = {
                    'mu': out['latent_list'][0][1][j].detach().cpu().numpy(),
                    'sigma': out['latent_list'][0][2][j].detach().cpu().numpy(),
                }
            # breakpoint()
            np.savez_compressed(os.path.join(out_dir, frame_name), **pred_info)

            # save_viz = (idx * batch_size + j) % 10
            save_viz=1
            if save_viz:
                trimesh.PointCloud(inp_verts_orig[j].detach().cpu().numpy()).export(os.path.join(out_dir_viz, frame_name+"_input.obj"))
                trimesh.PointCloud(inp_verts[j].detach().cpu().numpy()).export(os.path.join(out_dir_viz, frame_name+"_input_norm.obj"))
                trimesh.PointCloud(out['x_0_pred'][j].detach().cpu().numpy()).export(os.path.join(out_dir_viz, frame_name+"_out.obj"))

    print(f"Saved results to {out_dir_base}")