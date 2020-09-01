#!/bin/bash
#SBATCH --job-name=theta_sat_w1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=1:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar

. /etc/profile
source $HOME/.bashrc

cd /mnt/clemente/lensing/RodriguezGroups/codes_RGroups/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate py2env

srun python -u forGroup_qprofile.py -sample $SLURM_JOB_NAME -theta $SLURM_JOB_NAME -ncores 50
