import sys
sys.path.append('/mnt/clemente/lensing/lens_codes_v3.7')
sys.path.append('/home/eli/lens_codes_v3.7')
import numpy as np
from pylab import *
from multipoles_shear import *
import emcee
import time
from multiprocessing import Pool
import argparse
from astropy.io import fits

folder = '/mnt/clemente/lensing/redMaPPer/compressed/'
parser = argparse.ArgumentParser()
parser.add_argument('-file', action='store', dest='file_name', default='profile.fits')
parser.add_argument('-ncores', action='store', dest='ncores', default=4)
args = parser.parse_args()

file_name = args.file_name
ncores    = args.ncores
ncores    = int(ncores)

print('fitting monopole misscentred')
print(folder)
print(file_name)

profile = fits.open(folder+file_name)
h       = profile[1].header
p       = profile[1].data
zmean   = h['Z_MEAN']    
Mguess  = 10**h['lM200_NFW']*1.3

    

def log_likelihood(data_model, r, Gamma, e_Gamma):
    log_M200,pcc = data_model
    M200 = 10**log_M200
    multipoles = multipole_shear_parallel(r,M200=M200,misscentred = True,
                                ellip=0,z=zmean,components = ['t'],
                                verbose=False,ncores=ncores)
    model = model_Gamma(multipoles,'t', misscentred = True, pcc = pcc)
    sigma2 = e_Gamma**2
    return -0.5 * np.sum((Gamma - model)**2 / sigma2 + np.log(2.*np.pi*sigma2))
    

def log_probability(data_model, r, Gamma, e_Gamma):
    log_M200, pcc = data_model
    if 12.5 < log_M200 < 15.5 and 0.3 < pcc < 1.0:
        return log_likelihood(data_model, r, Gamma, e_Gamma)
    return -np.inf

# initializing

pos = np.array([np.random.uniform(12.5,15.5,10),
                np.random.normal(0.8,0.3,10)]).T

pccdist = pos[:,1]                
pos[pccdist > 1.,1] = 1.

nwalkers, ndim = pos.shape

#-------------------
# running emcee

#pool = Pool(processes=(ncores))



t1 = time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(p.Rp,p.DSigma_T,p.error_DSigma_T))
sampler.run_mcmc(pos, 450, progress=True)
print('//////////////////////')
print('         TIME         ')
print('----------------------')
print((time.time()-t1)/60.    )

# saving mcmc out

mcmc_out = sampler.get_chain(flat=True)

f1=open(folder+'monopole_misscentred_'+file_name,'w')
f1.write('# log(M200)  pcc \n')
np.savetxt(f1,mcmc_out,fmt = ['%12.6f']*2)
f1.close()


mcmc1 = (mcmc_out.T)[:,1500:]
lM200,pcc = np.median(mcmc1,axis=1)

h.append(('lM200',np.round(lM200,4)))
h.append(('pcc',np.round(pcc,4)))

fits.writeto(folder+file_name, p, h, overwrite=True)

