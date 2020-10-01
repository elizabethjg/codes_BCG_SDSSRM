#!/bin/bash
#SBATCH --job-name=total
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar

. /etc/profile
source $HOME/.bashrc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate py3env

cd /mnt/clemente/lensing/redMaPPer/compressed

srun python /mnt/clemente/lensing/codes_BCG_SDSSRM/fit_profile_monopole_misscentred.py -file 'profile_'$SLURM_JOB_NAME'.fits' -ncores $SLURM_CPUS_PER_TASK
