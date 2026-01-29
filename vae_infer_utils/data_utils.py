import os
import numpy as np
import trimesh

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

def gather_custom_data(in_fp, out_fp):
    all_verts = []
    out_fps = []

    obj = trimesh.load(in_fp)
    all_verts.append(obj.vertices)
    out_fps.append(out_fp)
    
    return all_verts, out_fps

def gather_data(data_name, in_fp=None, out_fp=None):
    DATASET_REGISTRY = {
        'ho3d_2048': {
            'data_dir': '/scratch/clear/atiwari/datasets/ho3d_v3_processing/models_sampled/verts_2048/',
            'out_dir': '/scratch/clear/atiwari/datasets/ho3d_v3_processing/lion_output/verts_2048/'
        },
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