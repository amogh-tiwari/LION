#!/bin/bash
#SBATCH --job-name=lion_regular                   # name of job
#SBATCH --output=lion_regular_%j.out              # output file (%j = job ID)
#SBATCH --error=lion_regular_%j.err               # error file (%j = job ID)
#SBATCH --account=tuy@v100                      # V100 accounting
#SBATCH --gres=gpu:4                            # reserve 4 GPUs
#SBATCH --cpus-per-task=24                      # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=16:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --partition=gpu_p2

module purge                                    # purge modules inherited by default
conda deactivate                                # deactivate environments inherited by default

module load miniforge/24.9.0
module load cuda/11.2
module load gcc/10.1.0

conda activate lion_env

cd $HOME/projects/LION

set -x                                          # activate echo of launched commands

echo "$SLURM_ARRAY_TASK_ID"

srun bash script/train_vae_all.sh 4 resume ./exp/1209/all/1c0883h_hvae_lion_B8/checkpoints/snapshot

