#!/usr/bin/env bash
#
# Slurm GPU Job Submission Script
# ================================
# This script tells the HPC system how to run your GPU programme
#
# Lines starting with #SBATCH are special instructions (directives) to Slurm
# (the job scheduling system). They must come before any commands.
#
# job configuration
# -----------------
#SBATCH --job-name=gpu_hello     # name that appears in queue
#SBATCH --partition=cs05r        # partition to be used
#SBATCH --time=00:05:00          # maximum runtime (HH:MM:SS)
                                 # job will be killed after this time
#SBATCH --nodes=1                # number of compute nodes
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # cpu cores per task
#SBATCH --gpus-per-task=1        # number of gpus per task
#SBATCH --mem=16G                # total memory per node
                                 # use G for gigabytes, M for megabytes

# the actual job starts here
# print job information
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"

# this provides python with cupy and cuda support
module load python/cuda12

# verify gpu is available
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# run your gpu programme
# ----------------------
# replace 'gpu_hello_world.py' with your script name
python gpu_hello_world.py
