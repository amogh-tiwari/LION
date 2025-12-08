# Setup

---
Option-1 (not tested): 
1. Create conda env using the yaml file: `conda env create -f env.yaml`
Then, jump to step-3.

Option-2 (tested)
1. Create conda env with python 3.8.12: `conda create -n <env_name> python=3.8.12"
2. Update the conda packages based on the env yaml file: `conda env update -n <env_name> -f env.yaml``

- - -
3. If above command exits succesfully, great -- skip the next steps below. If the above command exits with an error in the pip requirements (my case), do the below. 
4. Degrade setuptools as follows: ``.
5. Go to the pip_requirements file, and comment out the line corresponding to setuptools installation.
6. Install remaining pip requirements: `pip install -r pip_requirements.txt`.
---

(Optional, but recommended): Run `python build_pkg.py`. Some libraries used in the project build some cuda libraries at run-time (and then caches them on the machine). This script simply imports those libraries and allows building of those cuda packages. 
If there are problems in the cuda build, look for version issues with cuda/gcc. For **jean-zay**, i solved my problems by doing the below before running the `python build_pkg.py` command. 

1. Run a jean-zay node with sufficient cpu cores (10 in my case), to ensure sufficient ram allocation (jean zay's ram allocation is currently tied to cpu cores allotted): `srun --ntasks=1 --gres=gpu:1 --cpus-per-task=10 --account tuy@v100 --time=1:30:00 --pty bash`
2. Load right modules: `module load cuda/11.2` (11.1 would've been the exact match, but that wasn't available) and `module load gcc/10.1.0`.
3. export MAX_JOBS=1 (helps with reducing parallelization, so more ram available per each cuda pkg build)
4. python build_pkg.py
Sometimes (happenned once to me), it can get stuck at the start before even starting to build pkgs. In that case, try clearing the cache from previous failed build attempts: `rm -rf ~/.cache/torch_extensions/*`. Not sure why (or even if) it helps, but it did solve the problem for me, and can't harm for sure. 

