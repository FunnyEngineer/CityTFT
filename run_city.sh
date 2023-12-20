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

#SBATCH -J CitySim           # Job name
#SBATCH -o citydnn.out       # Name of stdout output file
#SBATCH -e citydnn.err       # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 8               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=funnyengineer@utexas.edu


# Launch serial code...
for URBAN_FILE in $(ls ./data/random_urban/*)
do
    for CLI_FILE in $(ls ./data/climate/citydnn/*)
    do
        urban_nmae=$(basename "$URBAN_FILE" .xml)
        # source main.sh -x $URBAN_FILE -c $CLI_FILE -p ./new_cli/citydnn -n ./new_xml/citydnn -e ./export/citydnn
    done
done
