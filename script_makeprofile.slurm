#!/bin/bash
#SBATCH --job-name=bcg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=10:00:00
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

srun python -u forGroup_qprofile.py -sample 'pcut_bin1_pw1'  -theta 'theta_sat_w1'   -ncores 56 -plim 0.9 -l_max 27.982
srun python -u forGroup_qprofile.py -sample 'pcut_bin2_w1' -theta 'theta_sat_pw1'  -ncores 56 -plim 0.9 -l_min 27.982
srun python -u forGroup_qprofile.py -sample 'pcut_z1_pw1'  -theta 'theta_sat_w1'   -ncores 56 -plim 0.9 -z_max 0.313
srun python -u forGroup_qprofile.py -sample 'pcut_z2_w1' -theta 'theta_sat_pw1'  -ncores 56 -plim 0.9 -z_min 0.313
