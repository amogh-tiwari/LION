import os
import numpy as np
import trimesh 

from easydict import EasyDict as edict
from natsort import natsorted
from tqdm import tqdm

def main(cfg):
    seq_names = natsorted(os.listdir(cfg.data_dir))
    
    n_seqs = len(seq_names)
    cfg.max_seqs = n_seqs if (cfg.max_seqs == "all" or cfg.max_seqs > n_seqs) else cfg.max_seqs
    for idx, seq in enumerate(tqdm(seq_names)):
        if idx >= cfg.max_seqs:
            print(f"Hit max seqs limit ({cfg.max_seqs}. Exitting ...)")
            exit()
        print(f"Processing seq #{idx}/{n_seqs}: {seq}")
        all_split_file_names = {}
        for split in ['train', 'val', 'test']:
            print(f"... split: {split}")
            curr_dir = os.path.join(cfg.data_dir, seq, split)
            curr_split_file_names = os.listdir(curr_dir)
            for curr_fn in tqdm(curr_split_file_names):
                in_fp = os.path.join(curr_dir, curr_fn)
                verts = np.load(in_fp)                
                out_fp = in_fp.replace("ShapeNetCore.v2.PC15k", "ShapeNetCore.v2.PC15k_visualizations/").replace(".npy", ".obj")
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)
                _ = trimesh.PointCloud(verts).export(out_fp)
            all_split_file_names[split] = curr_split_file_names
            breakpoint()

if __name__ == "__main__":
    cfg = {
        'data_dir': '/scratch/clear/atiwari/lion/data/ShapeNetCore.v2.PC15k',
        'max_seqs': 1
    }
    cfg = edict(cfg)
    main(cfg)