#!/bin/bash
#SBATCH --job-name=Mbin1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2-00:00
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ejgonzalez@unc.edu.ar

cd /home/elizabeth/profiles_redMapper/

export OMP_NUM_THREADS=10

module load softwares/anaconda2/5.0-intel-2018.0
source activate py2env


srun python /home/elizabeth/codes_BCG_SDSSRM/fit_profile_monopole_misscentred.py -file 'profile_'$SLURM_JOB_NAME'.fits' -ncores $SLURM_CPUS_PER_TASK
