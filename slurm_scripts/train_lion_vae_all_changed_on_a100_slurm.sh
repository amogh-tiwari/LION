#!/bin/bash
#SBATCH --job-name=lion_changed                  # name of job
#SBATCH --output=lion_changed_%j.out              # output file (%j = job ID)
#SBATCH --error=lion_changed_%j.err               # error file (%j = job ID)
#SBATCH --account=tuy@a100                      # A100 accounting
#SBATCH --gres=gpu:4                            # reserve 4 GPUs
#SBATCH --cpus-per-task=40                      # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=20:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --partition=gpu_p5
#SBATCH -C a100

module purge                                    # purge modules inherited by default
conda deactivate                                # deactivate environments inherited by default

module load arch/a100
module load miniforge/24.9.0
module load cuda/11.2
module load gcc/10.1.0

conda activate lion_env

cd $HOME/projects/LION

set -x                                          # activate echo of launched commands

echo "$SLURM_ARRAY_TASK_ID"

srun bash script/train_vae_all_changed.sh 4 resume exp/changed/0114/all/376fc0h_hvae_lion_B16/checkpoints/snapshot
