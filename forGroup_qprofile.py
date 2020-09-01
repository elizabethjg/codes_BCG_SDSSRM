import sys
sys.path.append('/mnt/clemente/lensing')
sys.path.append('/mnt/clemente/lensing/lens_codes_v3.7')
import time
import numpy as np
from lensing37 import gentools
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
import pandas as pd
from maria_func import *
from profiles_fit import *
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from multiprocessing import Pool
from multiprocessing import Process
import argparse

#parameters
cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

folder = '/mnt/clemente/lensing/redMaPPer/compressed/'
f      = fits.open(folder+'gx_redMapper.fits')
S = Table(f[2].data).to_pandas()  
S.set_index('CATID', inplace=True)


def partial_profile(backcat_ids,RA0,DEC0,Z,pangle,
                    RIN,ROUT,ndots,nboot=100):

        
        
        backcat     = S.loc[backcat_ids]
        backcat.Z_B = np.round(backcat.Z_B,2)
        ndots = int(ndots)
        
        if 'KiDS' in np.array(backcat.CATNAME)[0]:
                mask = (backcat.Z_B > (Z + 0.1))*(backcat.ODDS >= 0.5)*(backcat.Z_B < 0.9)
        else:
                mask = (backcat.Z_B > (Z + 0.1))*(backcat.ODDS >= 0.5)
                
        catdata = backcat[mask]


        dl, ds, dls = gentools.compute_lensing_distances(np.array([Z]), catdata.Z_B, precomputed=True)
        dl  = (dl/0.7)*h
        ds  = (ds/0.7)*h
        dls = (dls/0.7)*h
        
        KPCSCALE   = dl*(((1.0/3600.0)*np.pi)/180.0)*1000.0
        BETA_array = dls/ds
        
        Dl = dl*1.e6*pc
        sigma_c = (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)



        rads, theta, test1,test2 = eq2p2(np.deg2rad(catdata.RAJ2000),
                                        np.deg2rad(catdata.DECJ2000),
                                        np.deg2rad(RA0),
                                        np.deg2rad(DEC0))


        theta2 = (2.*np.pi - theta) +np.pi/2.
        theta_ra = theta2
        theta_ra[theta2 > 2.*np.pi] = theta2[theta2 > 2.*np.pi] - 2.*np.pi
               
        #Correct polar angle for e1, e2
        theta = theta+np.pi/2.
        
        e1     = catdata.e1
        e2     = catdata.e2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c
        
        del(e1)
        del(e2)
        
        r=np.rad2deg(rads)*3600*KPCSCALE
        del(rads)
        
        peso = catdata.weight
        peso = peso/(sigma_c**2) 
        m    = catdata.m
        
        Ntot = len(catdata)
        del(catdata)    
        
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        dig = np.digitize(r,bines)
        
        at     = theta_ra - pangle
        
        DSIGMAwsum_T = []
        DSIGMAwsum_X = []
        WEIGHTsum    = []
        Mwsum        = []
        
        BOOTwsum_T   = np.zeros((nboot,ndots))
        BOOTwsum_X   = np.zeros((nboot,ndots))
        BOOTwsum     = np.zeros((nboot,ndots))
        
        GAMMATcos_wsum = []
        GAMMAXsin_wsum = []
        WEIGHTcos_sum  = []
        WEIGHTsin_sum  = []
        
        BOOTwsum_Tcos  = np.zeros((nboot,ndots))
        BOOTwsum_Xsin  = np.zeros((nboot,ndots))
        BOOTwsum_cos   = np.zeros((nboot,ndots))
        BOOTwsum_sin   = np.zeros((nboot,ndots))
        
        GAMMATcos_wsum_c = []
        GAMMAXsin_wsum_c = []
        WEIGHTcos_sum_c  = []
        WEIGHTsin_sum_c  = []
        
        BOOTwsum_Tcos_c  = np.zeros((nboot,ndots))
        BOOTwsum_Xsin_c  = np.zeros((nboot,ndots))
        BOOTwsum_cos_c   = np.zeros((nboot,ndots))
        BOOTwsum_sin_c   = np.zeros((nboot,ndots))        
        
        for nbin in range(ndots):
                mbin = dig == nbin+1              
                
                DSIGMAwsum_T = np.append(DSIGMAwsum_T,(et[mbin]*peso[mbin]).sum())
                DSIGMAwsum_X = np.append(DSIGMAwsum_X,(ex[mbin]*peso[mbin]).sum())
                WEIGHTsum    = np.append(WEIGHTsum,(peso[mbin]).sum())
                Mwsum        = np.append(Mwsum,(m[mbin]*peso[mbin]).sum())

                GAMMATcos_wsum = np.append(GAMMATcos_wsum,(et[mbin]*np.cos(2.*at[mbin])*peso[mbin]).sum())
                GAMMAXsin_wsum = np.append(GAMMAXsin_wsum,(ex[mbin]*np.sin(2.*at[mbin])*peso[mbin]).sum())
                WEIGHTcos_sum  = np.append(WEIGHTcos_sum,((np.cos(2.*at[mbin])**2)*peso[mbin]).sum())
                WEIGHTsin_sum  = np.append(WEIGHTsin_sum,((np.sin(2.*at[mbin])**2)*peso[mbin]).sum())
                               
                GAMMATcos_wsum_c = np.append(GAMMATcos_wsum_c,(et[mbin]*np.cos(2.*theta_ra[mbin])*peso[mbin]).sum())
                GAMMAXsin_wsum_c = np.append(GAMMAXsin_wsum_c,(ex[mbin]*np.sin(2.*theta_ra[mbin])*peso[mbin]).sum())
                WEIGHTcos_sum_c  = np.append(WEIGHTcos_sum_c,((np.cos(2.*theta_ra[mbin])**2)*peso[mbin]).sum())
                WEIGHTsin_sum_c  = np.append(WEIGHTsin_sum_c,((np.sin(2.*theta_ra[mbin])**2)*peso[mbin]).sum())
                
                index = np.arange(mbin.sum())
                if mbin.sum() == 0:
                        continue
                else:
                        with NumpyRNGContext(1):
                                bootresult = bootstrap(index, nboot)
                        INDEX=bootresult.astype(int)
                        BOOTwsum_T[:,nbin] = np.sum(np.array(et[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_X[:,nbin] = np.sum(np.array(ex[mbin]*peso[mbin])[INDEX],axis=1)
                        BOOTwsum[:,nbin]   = np.sum(np.array(peso[mbin])[INDEX],axis=1)

                        BOOTwsum_Tcos[:,nbin] = np.sum(np.array(et[mbin]*np.cos(2.*at[mbin])*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_Xsin[:,nbin] = np.sum(np.array(ex[mbin]*np.sin(2.*at[mbin])*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_cos[:,nbin]  = np.sum(np.array(peso[mbin]*(np.cos(2.*at[mbin])**2))[INDEX],axis=1)
                        BOOTwsum_sin[:,nbin]  = np.sum(np.array(peso[mbin]*(np.sin(2.*at[mbin])**2))[INDEX],axis=1)

                        BOOTwsum_Tcos_c[:,nbin] = np.sum(np.array(et[mbin]*np.cos(2.*theta_ra[mbin])*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_Xsin_c[:,nbin] = np.sum(np.array(ex[mbin]*np.sin(2.*theta_ra[mbin])*peso[mbin])[INDEX],axis=1)
                        BOOTwsum_cos_c[:,nbin]  = np.sum(np.array(peso[mbin]*(np.cos(2.*theta_ra[mbin])**2))[INDEX],axis=1)
                        BOOTwsum_sin_c[:,nbin]  = np.sum(np.array(peso[mbin]*(np.sin(2.*theta_ra[mbin])**2))[INDEX],axis=1)

        
        output = {'DSIGMAwsum_T':DSIGMAwsum_T,'DSIGMAwsum_X':DSIGMAwsum_X,
                   'WEIGHTsum':WEIGHTsum, 'Mwsum':Mwsum, 
                   'BOOTwsum_T':BOOTwsum_T, 'BOOTwsum_X':BOOTwsum_X, 'BOOTwsum':BOOTwsum,
                   'GAMMATcos_wsum': GAMMATcos_wsum, 'GAMMAXsin_wsum': GAMMAXsin_wsum,
                   'WEIGHTcos_sum': WEIGHTcos_sum, 'WEIGHTsin_sum': WEIGHTsin_sum,
                   'GAMMATcos_wsum_c': GAMMATcos_wsum_c, 'GAMMAXsin_wsum_c': GAMMAXsin_wsum_c,
                   'WEIGHTcos_sum_c': WEIGHTcos_sum_c, 'WEIGHTsin_sum_c': WEIGHTsin_sum_c,
                   'BOOTwsum_Tcos':BOOTwsum_Tcos, 'BOOTwsum_Xsin':BOOTwsum_Xsin, 
                   'BOOTwsum_cos':BOOTwsum_cos, 'BOOTwsum_sin':BOOTwsum_sin, 
                   'BOOTwsum_Tcos_c':BOOTwsum_Tcos, 'BOOTwsum_Xsin_c':BOOTwsum_Xsin, 
                   'BOOTwsum_cos_c':BOOTwsum_cos, 'BOOTwsum_sin_c':BOOTwsum_sin, 
                   'Ntot':Ntot}
        
        return output

def partial_profile_unpack(minput):
	return partial_profile(*minput)
        

def main(sample='pru',l_min=20.,l_max=150.,
                z_min = 0.1, z_max = 0.4,
                RIN = 100., ROUT =5000.,
                proxy_angle = 'theta_sat_w1',
                ndots= 10,ncores=10,h=0.7):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        l_min          (int) lower limit of galaxy members - >=
        l_max          (int) higher limit of galaxy members - <
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        proxy_angle    (str) proxy definition of the angle to compute the quadrupole
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        h              (float) H0 = 100.*h
        '''

        cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)
        tini = time.time()
        
        print('Sample ',sample)
        print('Selecting groups with:')
        print(l_min,' <= Lambda < ',l_max)
        print(z_min,' <= z < ',z_max)
        print('Profile has ',ndots,'bins')
        print('from ',RIN,'kpc to ',ROUT,'kpc')
        print('Angle proxy ',proxy_angle)
        print('h ',h)
              
        # Defining radial bins
        bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)*1.e-3
        
        #reading cats
        
        L = Table(f[1].data).to_pandas()  
        angles = fits.open(folder+'SAT_angles.fits')[1].data
        borderid = np.loadtxt(folder+'redMapperID_border.list')
        
        zlambda = L.Z_LAMBDA
        zspec   = L.Z_SPEC
        Z_c      = zspec
        Z_c[Z_c<0] = zlambda[Z_c<0]
        L.Z_LAMBDA = Z_c
        
        mrich   = (L.LAMBDA >= l_min)*(L.LAMBDA < l_max)
        mz      = (L.Z_LAMBDA >= z_min)*(L.Z_LAMBDA < z_max)
        mborder = (~np.in1d(L.ID,borderid))
        mlenses = mrich*mz*mborder
        Nlenses = mlenses.sum()

        if Nlenses < ncores:
                ncores = Nlenses
        
        print('Nlenses',Nlenses)
        print('CORRIENDO EN ',ncores,' CORES')

        # A = fits.open('/mnt/clemente/lensing/RodriguezGroups/angle_Rgroups_FINAL.fits')[1].data
        # theta  = A.theta[mlenses]
        
        L = L[mlenses]
        theta  = angles[mlenses][proxy_angle]
        
        # SPLIT LENSING CAT
        
        lbins = int(round(Nlenses/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nlenses)]
        Lsplit = np.split(L.iloc[:],slices)
        Tsplit = np.split(theta,slices)
        
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        DSIGMAwsum_T = np.zeros(ndots) 
        DSIGMAwsum_X = np.zeros(ndots)
        WEIGHTsum    = np.zeros(ndots)
        Mwsum        = np.zeros(ndots)
        
        BOOTwsum_T   = np.zeros((100,ndots))
        BOOTwsum_X   = np.zeros((100,ndots))
        BOOTwsum     = np.zeros((100,ndots))
        
        GAMMATcos_wsum = np.zeros(ndots)
        GAMMAXsin_wsum = np.zeros(ndots)
        WEIGHTcos_sum  = np.zeros(ndots)
        WEIGHTsin_sum  = np.zeros(ndots)
        
        BOOTwsum_Tcos  = np.zeros((100,ndots))
        BOOTwsum_Xsin  = np.zeros((100,ndots))
        BOOTwsum_cos   = np.zeros((100,ndots))
        BOOTwsum_sin   = np.zeros((100,ndots))
        
        GAMMATcos_wsum_c = np.zeros(ndots)
        GAMMAXsin_wsum_c = np.zeros(ndots)
        WEIGHTcos_sum_c  = np.zeros(ndots)
        WEIGHTsin_sum_c  = np.zeros(ndots)
        
        BOOTwsum_Tcos_c  = np.zeros((100,ndots))
        BOOTwsum_Xsin_c  = np.zeros((100,ndots))
        BOOTwsum_cos_c   = np.zeros((100,ndots))
        BOOTwsum_sin_c   = np.zeros((100,ndots))        
        
        Ntot         = []
        tslice       = np.array([])
        
        for l in range(len(Lsplit)):
                
                print('RUN ',l+1,' OF ',len(Lsplit))
                
                t1 = time.time()
                
                num = len(Lsplit[l])
                
                rin  = RIN*np.ones(num)
                rout = ROUT*np.ones(num)
                nd   = ndots*np.ones(num)
                
                if num == 1:
                        entrada = [Lsplit[l].CATID.iloc[0],Lsplit[l].RA.iloc[0],
                                        Lsplit[l].DEC.iloc[0],Lsplit[l].Z_LAMBDA.iloc[0],
                                        Tsplit[l][0],RIN,ROUT,ndots]
                        
                        salida = [partial_profile_unpack(entrada)]
                else:          
                        entrada = np.array([Lsplit[l].CATID.iloc[:],Lsplit[l].RA,
                                        Lsplit[l].DEC,Lsplit[l].Z_LAMBDA,Tsplit[l][:],
                                        rin,rout,nd]).T
                        
                        pool = Pool(processes=(num))
                        salida = np.array(pool.map(partial_profile_unpack, entrada))
                        pool.terminate()
                                
                for profilesums in salida:
                                                
                        DSIGMAwsum_T += profilesums['DSIGMAwsum_T']
                        DSIGMAwsum_X += profilesums['DSIGMAwsum_X']
                        WEIGHTsum    += profilesums['WEIGHTsum']
                        Mwsum        += profilesums['Mwsum']
                        
                        BOOTwsum_T   += profilesums['BOOTwsum_T']
                        BOOTwsum_X   += profilesums['BOOTwsum_X']
                        BOOTwsum     += profilesums['BOOTwsum']
                        
                        GAMMATcos_wsum += profilesums['GAMMATcos_wsum']
                        GAMMAXsin_wsum += profilesums['GAMMAXsin_wsum']
                        WEIGHTcos_sum  += profilesums['WEIGHTcos_sum'] 
                        WEIGHTsin_sum  += profilesums['WEIGHTsin_sum'] 
                        
                        BOOTwsum_Tcos  += profilesums['BOOTwsum_Tcos']
                        BOOTwsum_Xsin  += profilesums['BOOTwsum_Xsin']
                        BOOTwsum_cos   += profilesums['BOOTwsum_cos']
                        BOOTwsum_sin   += profilesums['BOOTwsum_sin'] 
                        
                        GAMMATcos_wsum_c += profilesums['GAMMATcos_wsum_c']
                        GAMMAXsin_wsum_c += profilesums['GAMMAXsin_wsum_c']
                        WEIGHTcos_sum_c  += profilesums['WEIGHTcos_sum_c'] 
                        WEIGHTsin_sum_c  += profilesums['WEIGHTsin_sum_c'] 
                        
                        BOOTwsum_Tcos_c  += profilesums['BOOTwsum_Tcos_c']
                        BOOTwsum_Xsin_c  += profilesums['BOOTwsum_Xsin_c']
                        BOOTwsum_cos_c   += profilesums['BOOTwsum_cos_c']
                        BOOTwsum_sin_c   += profilesums['BOOTwsum_sin_c'] 
                        
                        Ntot         = np.append(Ntot,profilesums['Ntot'])
                
                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice,ts)
                print('TIME SLICE')
                print(ts)
                print('Estimated ramaining time')
                print(np.mean(tslice)*(len(Lsplit)-(l+1)))
        
        # COMPUTING PROFILE        
                
        Mcorr     = Mwsum/WEIGHTsum
        DSigma_T  = (DSIGMAwsum_T/WEIGHTsum)/(1+Mcorr)
        DSigma_X  = (DSIGMAwsum_X/WEIGHTsum)/(1+Mcorr)
        eDSigma_T =  np.std((BOOTwsum_T/BOOTwsum),axis=0)/(1+Mcorr)
        eDSigma_X =  np.std((BOOTwsum_X/BOOTwsum),axis=0)/(1+Mcorr)
        
        GAMMA_Tcos = (GAMMATcos_wsum/WEIGHTcos_sum)/(1+Mcorr)
        GAMMA_Xsin = (GAMMAXsin_wsum/WEIGHTsin_sum)/(1+Mcorr)
        eGAMMA_Tcos =  np.std((BOOTwsum_Tcos/BOOTwsum_cos),axis=0)/(1+Mcorr)
        eGAMMA_Xsin =  np.std((BOOTwsum_Xsin/BOOTwsum_sin),axis=0)/(1+Mcorr)

        GAMMA_Tcos_c = (GAMMATcos_wsum_c/WEIGHTcos_sum_c)/(1+Mcorr)
        GAMMA_Xsin_c = (GAMMAXsin_wsum_c/WEIGHTsin_sum_c)/(1+Mcorr)
        eGAMMA_Tcos_c =  np.std((BOOTwsum_Tcos_c/BOOTwsum_cos_c),axis=0)/(1+Mcorr)
        eGAMMA_Xsin_c =  np.std((BOOTwsum_Xsin_c/BOOTwsum_sin_c),axis=0)/(1+Mcorr)

        
        # AVERAGE LENS PARAMETERS
        
        zmean        = np.average(L.Z_LAMBDA,weights=Ntot)
        l_mean       = np.average(L.LAMBDA,weights=Ntot)
        
        # FITING AN NFW MODEL
        
        H        = cosmo.H(zmean).value/(1.0e3*pc) #H at z_pair s-1 
        roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
        roc_mpc  = roc*((pc*1.0e6)**3.0)
        
        try:
                nfw        = NFW_stack_fit(R,DSigma_T,eDSigma_T,zmean,roc)
        except:
                nfw          = [0.01,0.,100.,[0.,0.],[0.,0.],-999.,0.]

        M200_NFW   = (800.0*np.pi*roc_mpc*(nfw[0]**3))/(3.0*Msun)
        e_M200_NFW =((800.0*np.pi*roc_mpc*(nfw[0]**2))/(Msun))*nfw[1]
        le_M200    = (np.log(10.)/M200_NFW)*e_M200_NFW
 
        # WRITING OUTPUT FITS FILE
        
        
        tbhdu = fits.BinTableHDU.from_columns(
                [fits.Column(name='Rp', format='D', array=R),
                fits.Column(name='DSigma_T', format='D', array=DSigma_T),
                fits.Column(name='error_DSigma_T', format='D', array=eDSigma_T),
                fits.Column(name='DSigma_X', format='D', array=DSigma_X),
                fits.Column(name='error_DSigma_X', format='D', array=eDSigma_X),
                fits.Column(name='GAMMA_Tcos', format='D', array=GAMMA_Tcos),
                fits.Column(name='error_GAMMA_Tcos', format='D', array=eGAMMA_Tcos),
                fits.Column(name='GAMMA_Xsin', format='D', array=GAMMA_Xsin),
                fits.Column(name='error_GAMMA_Xsin', format='D', array=eGAMMA_Xsin),
                fits.Column(name='GAMMA_Tcos_c', format='D', array=GAMMA_Tcos_c),
                fits.Column(name='error_GAMMA_Tcos_c', format='D', array=eGAMMA_Tcos_c),
                fits.Column(name='GAMMA_Xsin_c', format='D', array=GAMMA_Xsin_c),
                fits.Column(name='error_GAMMA_Xsin_c', format='D', array=eGAMMA_Xsin_c)])
        
        h = tbhdu.header
        h.append(('N_LENSES',np.int(Nlenses)))
        h.append(('l_min',np.int(l_min)))
        h.append(('l_max',np.int(l_max)))
        h.append(('z_min',np.round(z_min,4)))
        h.append(('z_max',np.round(z_max,4)))
        h.append(('lM200_NFW',np.round(np.log10(M200_NFW),4)))
        h.append(('elM200_NFW',np.round(le_M200,4)))
        h.append(('CHI2_NFW',np.round(nfw[2],4)))
        h.append(('l_mean',np.round(l_mean,4)))
        h.append(('z_mean',np.round(zmean,4)))
              
        
        tbhdu.writeto(folder+'profile_'+sample+'.fits',overwrite=True)
                
        tfin = time.time()
        
        print('TOTAL TIME ',(tfin-tini)/60.)
        


if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-sample', action='store', dest='sample',default='pru')
        parser.add_argument('-l_min', action='store', dest='l_min', default=20)
        parser.add_argument('-l_max', action='store', dest='l_max', default=150)
        parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
        parser.add_argument('-z_max', action='store', dest='z_max', default=0.4)
        parser.add_argument('-RIN', action='store', dest='RIN', default=100.)
        parser.add_argument('-ROUT', action='store', dest='ROUT', default=5000.)
        parser.add_argument('-theta', action='store', dest='theta', default='theta_sat_w1')
        parser.add_argument('-nbins', action='store', dest='nbins', default=10)
        parser.add_argument('-ncores', action='store', dest='ncores', default=10)
        parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=0.7)
        args = parser.parse_args()
        
        sample     = args.sample
        l_min      = float(args.l_min) 
        l_max      = float(args.l_max) 
        z_min      = float(args.z_min) 
        z_max      = float(args.z_max) 
        RIN        = float(args.RIN)
        ROUT       = float(args.ROUT)
        theta      = args.theta
        nbins      = int(args.nbins)
        ncores     = int(args.ncores)
        h          = float(args.h_cosmo)
        
        main(sample,l_min,l_max, z_min, z_max, RIN, ROUT,theta,nbins,ncores,h)
