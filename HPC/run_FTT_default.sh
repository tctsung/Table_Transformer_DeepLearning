#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ct2840@nyu.edu
#SBATCH --job-name=FTT_default
#SBATCH --output=slurm_%j_FTT_default.out

module purge

singularity exec --nv\
	    --overlay /scratch/ct2840/env/my_pytorch38.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python resnet_runner.py > checklog.out"
