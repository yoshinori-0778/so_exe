#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --qos=tiger-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=100GB
#SBATCH --job-name=atm_ratio_test  # Name of the job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ys5857@princeton.edu

set -e

# Only needed if using the planets module to access planet data from web
#export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file for capturing the output of the script
log="./atm_ratio_test.log"

#=========== Compute runtime parameters ============
echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

# Set the number of OpenMP threads per process
export OMP_NUM_THREADS=1

# Define the SLURM launch command for parallel execution
launch_str="srun --export=ALL"

# Command to execute the Python script with the required arguments
com="${launch_str} python3 /home/ys5857/workspace/script/so_exe/atm/atm_ratio.py \
    --test"
    

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"