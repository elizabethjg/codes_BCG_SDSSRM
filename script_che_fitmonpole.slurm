#!/bin/bash
#SBATCH --job-name=total
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar

cd /home/elizabeth/profiles_redMapper/

export OMP_NUM_THREADS=10

module load softwares/anaconda2/5.0-intel-2018.0
source activate py3env

srun python fit_profile_monopole_misscentred.py -file 'profile_'$SLURM_JOB_NAME'.fits' -ncores $SLURM_CPUS_PER_TASK
