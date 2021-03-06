import sys
sys.path.append('/mnt/clemente/lensing/lens_codes_v3.7')
sys.path.append('/home/elizabeth/lens_codes_v3.7')
import numpy as np
from pylab import *
from multipoles_shear import *
import emcee
import time
from multiprocessing import Pool
import argparse
from astropy.io import fits

# folder = '/mnt/clemente/lensing/redMaPPer/compressed/'
folder = './'

parser = argparse.ArgumentParser()
parser.add_argument('-file', action='store', dest='file_name', default='profile.fits')
parser.add_argument('-ncores', action='store', dest='ncores', default=15)
parser.add_argument('-misscentred', action='store',
                    dest='miss', default='False')
parser.add_argument('-component', action='store',
                    dest='component', default='both')
parser.add_argument('-RIN', action='store', dest='RIN', default=0)
parser.add_argument('-ROUT', action='store', dest='ROUT', default=5000)
parser.add_argument('-nit', action='store', dest='nit', default=250)
parser.add_argument('-continue', action='store', dest='cont', default='False')
args = parser.parse_args()

file_name = args.file_name

if 'True' in args.miss:
	miss = True
elif 'False' in args.miss:
	miss = False

if 'True' in args.cont:
	cont = True
elif 'False' in args.cont:
	cont = False


component = args.component
nit = int(args.nit)
ncores = args.ncores
ncores = int(ncores)
rin = float(args.RIN)
rout = float(args.ROUT)

if miss:
	outfile = folder+'quadrupole_'+component+'_miss_' + \
	    file_name[:-5]+'_'+str(int(rin))+'_'+str(int(rout))+'.out'
	backup = folder+'backup_bcg_'+component+'_miss_' + \
	    file_name[:-5]+'_'+str(int(rin))+'_'+str(int(rout))+'.out'
else:
	outfile = folder+'quadrupole_'+component+'_' + \
	    file_name[:-5]+'_'+str(int(rin))+'_'+str(int(rout))+'.out'
	backup = folder+'backup_bcg_'+component+'_'+file_name[:-5]+ \
	    '_'+str(int(rin))+'_'+str(int(rout))+'.out'


print('fitting quadrupole')
print(file_name)
print('miss = ', miss)
print(component)
print('ncores = ', ncores)
print('RIN ', rin)
print('ROUT ', rout)
print('nit', nit)
print('continue', cont)
print('outfile', outfile)


profile = fits.open(folder+file_name)
h       = profile[1].header
p       = profile[1].data
zmean   = h['Z_MEAN']    
M200    = 10**h['lM200']
pcc     = h['pcc']


print('M200', M200)
print('pcc', pcc)


def log_likelihood(data_model, r, Gamma, e_Gamma):
    ellip = data_model
    if 'both' in component:
        r = np.split(r, 2)[0]

        multipoles = multipole_shear_parallel(r, M200=M200, misscentred=miss,
                    ellip=ellip, z=zmean, components=['tcos', 'xsin'],
                    verbose=False, ncores=ncores)
        model_t = model_Gamma(multipoles, 'tcos', misscentred=miss, pcc=pcc)
        model_x = model_Gamma(multipoles, 'xsin', misscentred=miss, pcc=pcc)
        model = np.append(model_t, model_x)
    else:
        multipoles = multipole_shear_parallel(r, M200=M200, misscentred=miss,
                    ellip=ellip, z=zmean, components=[component],
                    verbose=False, ncores=ncores)
        model = model_Gamma(multipoles, component, misscentred=miss, pcc=pcc)
    sigma2 = e_Gamma**2
    return -0.5 * np.sum((Gamma - model)**2 / sigma2 + np.log(2.*np.pi*sigma2))


def log_probability(data_model, r, Gamma, e_Gamma):
    ellip = data_model
    if 0. < ellip < 0.5:
        return log_likelihood(data_model, r, Gamma, e_Gamma)
    return -np.inf

# initializing


pos = np.array([np.random.uniform(0., 0.5, 10)]).T

nwalkers, ndim = pos.shape

#-------------------
# running emcee

maskr = (p.Rp > (rin/1000.))*(p.Rp < (rout/1000.))
p = p[maskr]

t1 = time.time()

backend = emcee.backends.HDFBackend(backup)
if not cont:
    backend.reset(nwalkers, ndim)


if component == 'tcos':
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(p.Rp, p.GAMMA_Tcos, p.error_GAMMA_Tcos),
				    backend=backend)
elif component == 'xsin':
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(p.Rp, p.GAMMA_Xsin,p.error_GAMMA_Xsin),
                                    backend=backend)
elif component == 'both':
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(np.append(p.Rp, p.Rp),
                                    np.append(p.GAMMA_Tcos, p.GAMMA_Xsin),
                                    np.append(p.error_GAMMA_Tcos,p.error_GAMMA_Xsin)),
                                    backend=backend)


if cont:
    sampler.run_mcmc(None, nit, progress=True)
else:
    sampler.run_mcmc(pos, nit, progress=True)

print((time.time()-t1)/60.)

#-------------------
# saving mcmc out

mcmc_out=sampler.get_chain(flat=True)


f1=open(outfile, 'w')
f1.write('# ellip \n')
np.savetxt(f1, mcmc_out, fmt=['%12.6f'])
f1.close()
