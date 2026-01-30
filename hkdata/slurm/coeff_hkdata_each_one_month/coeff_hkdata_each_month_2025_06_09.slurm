#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --qos=tiger-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=coeff_hkdata_260128_each_month_2025_06_09 # Name of the job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ys5857@princeton.edu

# Log file for capturing the output of the script
log="./logs/coeff_hkdata_260128_each_month_2025_06_09.log"

#=========== Compute runtime parameters ============
echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

# Set the number of OpenMP threads per process
export OMP_NUM_THREADS=4

# Define the SLURM launch command for parallel execution
launch_str="srun --export=ALL"

# Command to execute the Python script with the required arguments
com="${launch_str} python3 /home/ys5857/workspace/script/so_exe/hkdata/coeff_hkdata.py \
    --nproc 0 --ys 2025 --ms 6 --ye 2025 --me 9"
    

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"