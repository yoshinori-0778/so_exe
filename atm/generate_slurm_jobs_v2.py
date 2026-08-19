import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_slurm_satp3(ys, ms, ye, me, outdir="slurm_jobs"):
    """
    月単位で ys, ye, ms, me をずらした SLURM script を生成
    """

    start = datetime(ys, ms, 1)
    end = datetime(ye, me, 1)

    os.makedirs(outdir, exist_ok=True)

    t = start
    while t <= end:
        ys2 = t.year
        ms2 = t.month

        # 月をまたぐ：翌月
        t_next = t + relativedelta(months=1)
        ye2 = t_next.year
        me2 = t_next.month

        # ファイル名
        fname = f"atm_ratio_satp3_{ys2}{ms2:02d}.slurm"
        fpath = os.path.join(outdir, fname)

        # SLURM script テンプレート
        content = f"""#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --qos=tiger-short
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=700GB
#SBATCH --job-name=satp3_atm_ratio_{ys2}{ms2:02d}
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ys5857@princeton.edu

log="./atm_ratio_satp3_{ys2}{ms2:02d}.log"

# Set the number of OpenMP threads per process
export OMP_NUM_THREADS=4

launch_str="srun --export=ALL"

com="${{launch_str}} python3 /home/ys5857/workspace/jupyter/2025/11/exe/atm_ratio.py \\
    --nproc 20 --sat satp3 --ys {ys2} --ye {ye2} --ms {ms2} --me {me2} --saved /scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp3"

echo $com
echo "Launching pipeline at $(date)"
eval $com > $log 2>&1
echo "Ending batch script at $(date)"
"""

        with open(fpath, "w") as f:
            f.write(content)

        print(f"Generated: {fpath}")

        # 次の月へ
        t = t_next

def generate_slurm_satp1(ys, ms, ye, me, outdir="slurm_jobs"):
    """
    月単位で ys, ye, ms, me をずらした SLURM script を生成
    """

    start = datetime(ys, ms, 1)
    end = datetime(ye, me, 1)

    os.makedirs(outdir, exist_ok=True)

    t = start
    while t <= end:
        ys2 = t.year
        ms2 = t.month

        # 月をまたぐ：翌月
        t_next = t + relativedelta(months=1)
        ye2 = t_next.year
        me2 = t_next.month

        # ファイル名
        fname = f"atm_ratio_satp1_{ys2}{ms2:02d}.slurm"
        fpath = os.path.join(outdir, fname)

        # SLURM script テンプレート
        content = f"""#!/bin/bash
#SBATCH --account=simonsobs
#SBATCH --qos=tiger-short
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=700GB
#SBATCH --job-name=satp1_atm_ratio_{ys2}{ms2:02d}
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ys5857@princeton.edu

log="./atm_ratio_satp1_{ys2}{ms2:02d}.log"

# Set the number of OpenMP threads per process
export OMP_NUM_THREADS=4

launch_str="srun --export=ALL"

com="${{launch_str}} python3 /home/ys5857/workspace/jupyter/2025/11/exe/atm_ratio.py \\
    --nproc 20 --sat satp1 --ys {ys2} --ye {ye2} --ms {ms2} --me {me2} --saved /scratch/gpfs/SIMONSOBS/users/ys5857/workspace/output/2026/06/atm_ratio/satp1"

echo $com
echo "Launching pipeline at $(date)"
eval $com > $log 2>&1
echo "Ending batch script at $(date)"
"""

        with open(fpath, "w") as f:
            f.write(content)

        print(f"Generated: {fpath}")

        # 次の月へ
        t = t_next

if __name__ == "__main__":
    generate_slurm_satp3(2024, 7, 2026, 6, outdir = 'slurm_jobs_satp3')
    generate_slurm_satp1(2024, 7, 2026, 6, outdir = 'slurm_jobs_satp1')
