#!/bin/bash
#OAR -n atiwari
#OAR -l host=1/gpuid=2,walltime=96:0:0
#OAR --array 1

source gpu_setVisibleDevices.sh

source "/scratch/clear/atiwari/miniconda3/etc/profile.d/conda.sh"
echo "Initialized miniconda"

conda activate lion_env
echo "Activated conda environment"
cd /home/atiwari/projects/LION/
echo Running Code ...

bash ./script/train_vae_all.sh 2
