#!/bin/bash
#SBATCH --job-name=TravailGPU                   # name of job
#SBATCH --output=TravailGPU_%j.out              # output file (%j = job ID)
#SBATCH --error=TravailGPU_%j.err               # error file (%j = job ID)
#SBATCH --constraint=v100-16g                   # reserve GPUs with 16 GB of RAM
#SBATCH --nodes=1                               # reserve 1 node
#SBATCH --ntasks=1                              # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:4                            # reserve 4 GPUs
#SBATCH --cpus-per-task=9                       # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=96:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4                        # QoS
#SBATCH --hint=nomultithread                    # deactivate hyperthreading
#SBATCH --account=tuy@v100                      # V100 accounting

module purge                                    # purge modules inherited by default
conda deactivate                                # deactivate environments inherited by default

module load miniforge/24.9.0
conda activate lion_env

cd $HOME/projects/LION

set -x                                          # activate echo of launched commands

echo "$SLURM_ARRAY_TASK_ID"

srun script/train_vae_all.sh 4
