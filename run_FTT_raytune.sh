#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1-3
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ct2840@nyu.edu
#SBATCH --job-name=RayTune_FTT
#SBATCH --output=S_RayTune%A_%a.out

module purge

singularity exec --nv\
	    --overlay /scratch/ct2840/env/my_pytorch38.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python FTT_ray_tune.py $SLURM_ARRAY_TASK_ID > checklog_%a.out"
