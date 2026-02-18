import os
import numpy as np
import trimesh

from glob import glob

def gather_ho3d_data(data_dir, out_dir):
    obj_dirs = os.listdir(data_dir)
    all_verts = []
    out_fps = []

    for obj_dir in obj_dirs:
        if not os.path.isdir(os.path.join(data_dir, obj_dir)): # Skip processing if it is not a dir
            continue
        obj_fp = os.path.join(data_dir, obj_dir, "textured_simple.obj")
        obj = trimesh.load(obj_fp)
        all_verts.append(obj.vertices)
        out_fps.append(os.path.join(out_dir, obj_dir))
    
    return all_verts, out_fps

def gather_custom_data(in_fps_all, out_fps_all):
    all_verts = []
    out_fps = []

    for idx, in_fp in enumerate(in_fps_all):
        obj = trimesh.load(in_fp)
        all_verts.append(obj.vertices)
        out_fps.append(out_fps_all[idx])
    return all_verts, out_fps

def gather_data(data_name, in_fp=None, out_fp=None):
    DATASET_REGISTRY = {
        'ho3d_2048': {
            'data_dir': '/scratch/clear/atiwari/datasets/ho3d_v3_processing/models_sampled/verts_2048/',
            'out_dir': '/scratch/clear/atiwari/datasets/ho3d_v3_processing/lion_output/verts_2048/'
        },
        # 'shape_net_3k': {
        #     'data_dir': '/scratch/clear/atiwari/datasets/ShapeNetCore.v2_subsetPC3K/ShapeNetCore.v2_subsetPC3K',
        #     'out_dir': '/scratch/clear/atiwari/datasets/ShapeNetCore.v2_lion_embeds/ShapeNetCore.v2_lion_embeds/'
        # },
        'custom': {
            'data_dir': '',
            'out_dir': '',
        }
    }
    assert data_name in DATASET_REGISTRY.keys(), f"data_name ({data_name}) must be one of DATASET_REGISTRY keys ({DATASET_REGISTRY.keys()})"

    data_dir = DATASET_REGISTRY[data_name]['data_dir']
    out_dir = DATASET_REGISTRY[data_name]['out_dir']

    if data_name == "ho3d_2048":
        all_pts, out_fps = gather_ho3d_data(data_dir, out_dir)
    
    if data_name == "custom":
        all_pts, out_fps = gather_custom_data(in_fp, out_fp)

    return np.asarray(all_pts), np.asarray(out_fps)

def gather_evaluation_assets(in_dir):
    # Just copied from object_manipulation directory.
    # in_dir = '../../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/grab_2048/'
    obj_names = os.listdir(in_dir)

    all_pcds = {}
    all_meshes = {}
    for obj_name in obj_names:
        curr_dir = os.path.join(in_dir, obj_name)
        if not os.path.isdir(curr_dir):
            continue

        all_meshes[obj_name] = glob(os.path.join(curr_dir, "*mesh_transform*"))
        # all_pcds[obj_name] = glob(os.path.join(curr_dir, "*pcd_transform*"))
        all_pcds[obj_name] = [p.replace("mesh_transform", "pcd_transform") for p in all_meshes[obj_name]]


    # Get all emenents of list to dict.        
    all_pcds_fps = []
    for key in all_pcds.keys():
        for fp in all_pcds[key]:
            all_pcds_fps.append(fp)
    return all_pcds_fps

if __name__ == "__main__":
    in_dir = '../../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/ho3d_2048/'
    gather_evaluation_assets(in_dir)
