#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --qos=tiger-short
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=vel_accl_satp1_multiprocess_260202 # Name of the job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ys5857@princeton.edu

# Log file for capturing the output of the script
log=".log/vel_accl_satp1_multiprocess_260202.log"

#=========== Compute runtime parameters ============
echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

# Define the SLURM launch command for parallel execution
launch_str="srun --export=ALL"

# Command to execute the Python script with the required arguments
com="${launch_str} python3 /home/ys5857/workspace/jupyter/2025/10/exe/vel_accl.py \
    2023 7 2026 2 --sat satp1 --nproc 14"
    

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"