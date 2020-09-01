#!/bin/bash
#SBATCH --job-name=bcg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=2:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar

. /etc/profile
source $HOME/.bashrc

cd /mnt/clemente/lensing/codes_BCG_SDSSRM/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate py3env

srun python -u forGroup_qprofile.py -sample 'bcg_tot'  -theta 'theta_bcg' -ncores 50
srun python -u forGroup_qprofile.py -sample 'bcg_bin1' -theta 'theta_bcg' -l_max 27.982 -ncores 50
srun python -u forGroup_qprofile.py -sample 'bcg_bin2' -theta 'theta_bcg' -l_min 27.982 -ncores 50
srun python -u forGroup_qprofile.py -sample 'bcg_z1'   -theta 'theta_bcg' -z_max 0.313  -ncores 50
srun python -u forGroup_qprofile.py -sample 'bcg_z2'   -theta 'theta_bcg' -z_min 0.313  -ncores 50
