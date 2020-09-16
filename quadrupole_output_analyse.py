import sys
sys.path.append('/home/eli/lens_codes_v3.7')
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from pylab import *
from multipoles_shear import *
import time
import os
from astropy.io import fits
# from profiles_fit import chi_red

folder2 = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/BCG_orientation/profiles/'


rin_bcg = np.array([0,0,700])
rout_bcg = np.array([5000,700,5000])

samples = ['total','bin1','bin2','z1','z2']

f1=open(folder2+'table_bcg.out','w')

for i in range(3):
	
	for s in samples:
	
		mcmc = np.loadtxt(folder2+'quadrupole_bcg_both_'+s+'_'+str(int(rin_bcg[i]))+'_'+str(int(rout_bcg[i]))+'.out')[1000:]
		p = np.percentile(mcmc, [16, 50, 84])
		plt.figure()
		plt.title(s)
		plt.plot(mcmc,'C1',alpha=0.5)
		plt.axhline(p[1])
		plt.axhline(p[2],ls='--')
		plt.axhline(p[0],ls='--')
		plt.savefig(folder2+'mcmc_'+s+'_'+str(int(rin_bcg[i]))+'_'+str(int(rout_bcg[i]))+'_out.png')
		
		
		f = open(folder2+'profile_total.cat', 'r')
		lines = f.readlines()
		j = lines[2].find('=')+1
		zmean = float(lines[2][j:-2])
		pcc = float((lines[-1][1:-2]))
		M200 = float((lines[-2][1:-2]))*1.e14
	
		
		profile = fits.open(folder2+'profile_bcg_'+s+'.fits')[1].data
		
		r  = np.logspace(np.log10(min(profile.Rp)),
						np.log10(max(profile.Rp)),10)
		
		multipoles = multipole_shear_parallel(r,M200=M200,
									ellip=p[1],z=zmean,
									verbose=False,ncores=2)
	
		f, ax = plt.subplots(1, 2, figsize=(8,5))
		ax[0].scatter(profile.Rp,profile.GAMMA_Tcos,facecolor='none',edgecolors='0.4')
		ax[0].plot(r,multipoles['Gt2'],'C4')
		ax[0].errorbar(profile.Rp,profile.GAMMA_Tcos,yerr=profile.error_GAMMA_Tcos,fmt = 'none',ecolor='0.4')
		ax[0].set_xscale('log')
		ax[0].set_yscale('log')
		ax[0].set_xlabel('R [mpc]')
		ax[0].set_ylim(0.1,200)
		ax[0].set_xlim(0.1,5.2)
		ax[0].xaxis.set_ticks([0.1,1,5])
		ax[0].set_xticklabels([0.1,1,5])
		ax[0].yaxis.set_ticks([1,10,100])
		ax[0].set_yticklabels([1,10,100])
		plt.legend()
		
		ax[1].scatter(profile.Rp,profile.GAMMA_Xsin,facecolor='none',edgecolors='0.4')
		ax[1].plot(r,multipoles['Gx2'],'C4')
		ax[1].errorbar(profile.Rp,profile.GAMMA_Xsin,yerr=profile.error_GAMMA_Xsin,fmt = 'none',ecolor='0.4')
		ax[1].set_xlabel('R [mpc]')
		ax[1].set_xscale('log')
		ax[1].set_ylim(-50,50)
		ax[1].set_xlim(0.1,5.2)
		ax[1].xaxis.set_ticks([0.1,1,5])
		ax[1].set_xticklabels([0.1,1,5])
		ax[1].yaxis.set_ticks([-30,-15,0,15,30])
		ax[1].set_yticklabels([-30,-15,0,15,30])
		plt.legend()
		f.subplots_adjust(hspace=0,wspace=0)
		plt.savefig(folder2+'quadrupole_profile'+s+'_'+str(int(rin_bcg[i]))+'_'+str(int(rout_bcg[i]))+'.png')
	
			# #/////// save file ///////
	
		
		f1.write('bcg_'+s+'_'+str(int(rin_bcg[i]))+'_'+str(int(rout_bcg[i]))+'    ') 
		f1.write(str('%.2f' % (M200/1.e14))+'   '+str('%.2f' % (zmean))+'   '+str(int(rin_bcg[i]))+'   '+str(int(rout_bcg[i]))+'    ')
		f1.write(str('%.2f' % (p[1]))+'   '+str('%.2f' % (np.diff(p)[0]))+'   '+str('%.2f' % (np.diff(p)[1]))+'    \n')

f1.close()

