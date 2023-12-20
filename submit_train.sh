#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch milan.serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J CityTFT           # Job name
#SBATCH -o CityTFT.out       # Name of stdout output file
#SBATCH -e CityTFT.err       # Name of stderr error file
#SBATCH -p gpu-a100-small          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -A MSS23005
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=funnyengineer@utexas.edu

# Any other commands must follow all #SBATCH directives...
# module load cuda/12.0 cudnn/8.8.1 nccl/2.11.4
source /work/08388/tudai/ls6/envs/mae/bin/activate

# Launch serial code...
python3 src/12_multi_urban_train.py
