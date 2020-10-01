import sys
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70., Om0=0.3, Ode0=0.7)


clusters_full = fits.open('/mnt/clemente/lensing/redMaPPer/redmapper_dr8_public_v6.3_catalog_Expanded.fits')[1].data
clusters = fits.open('/mnt/clemente/lensing/redMaPPer/compressed/gx_redMapper.fits')[1].data
ang_sat  = fits.open('/mnt/clemente/lensing/redMaPPer/angles_redMapper_forprofile.fits')[1].data
IDang = fits.open('/mnt/clemente/lensing/redMaPPer/redmapper_dr8_public_v6.3_catalog.fits')[1].data.ID

IDc    = clusters.ID 
IDf    = clusters_full.ID 
mid    = np.in1d(IDang,IDc)
sindex = np.argsort(IDc)

# DISTANCIA AL VECINO

RA  = clusters.RA
DEC = clusters.DEC
z   = clusters.Z_LAMBDA

Dcosmo = cosmo.comoving_distance(z)
catalog = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=Dcosmo)
idx, d2d, d3d = catalog.match_to_catalog_3d(catalog, nthneighbor=2)

#--------------------------------

ides  = np.zeros(len(IDc))
t     = np.zeros(len(IDc))
t_wl  = np.zeros(len(IDc))
t_wd  = np.zeros(len(IDc))
t_p   = np.zeros(len(IDc))
t_pwl = np.zeros(len(IDc))
t_pwd = np.zeros(len(IDc))
pcen  = np.zeros(len(IDc))

t_BG  = clusters.deVPhi_BG
P_cen = (clusters_full.P_CEN).T[0]


mneg        = t_BG < 0.
t_BG[mneg]  = 360. + t_BG[mneg]  
t_BG        = (360. - t_BG) + 90.
mmas        = t_BG > 360.
t_BG[mmas]  = t_BG[mmas] - 360.
mmas        = t_BG > 180.
t_BG[mmas]  = t_BG[mmas] - 180.
t_BG        = np.deg2rad(t_BG)


pcen[sindex]  = P_cen[mid]
ides[sindex]  = IDang[mid]
t[sindex]     = ang_sat.theta[mid]
t_wl[sindex]  = ang_sat.theta_wlum[mid]
t_wd[sindex]  = ang_sat.theta_wd[mid]
t_p[sindex]   = ang_sat.theta_pcut[mid]
t_pwl[sindex] = ang_sat.theta_pcut_wlum[mid]
t_pwd[sindex] = ang_sat.theta_pcut_wd[mid]

tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='ID', format='K', array=ides),
        fits.Column(name='P_cen', format='D', array=pcen),
        fits.Column(name='Rprox', format='D', array=np.array(d3d)),
        fits.Column(name='theta_sat_w1', format='D', array=t),
        fits.Column(name='theta_sat_wl', format='D', array=t_wl),
        fits.Column(name='theta_sat_wd', format='D', array=t_wd),
        fits.Column(name='theta_sat_pw1', format='D', array=t_p),
        fits.Column(name='theta_sat_pwl', format='D', array=t_pwl),
        fits.Column(name='theta_sat_pwd', format='D', array=t_pwd),
        fits.Column(name='theta_bcg', format='D', array=t_BG)])

tbhdu.writeto('/mnt/clemente/lensing/redMaPPer/compressed/SAT_angles.fits',overwrite=True)        


clusters = fits.open('/mnt/clemente/lensing/redMaPPer/redmapper_dr8_public_v6.3_catalog_Expanded.fits')[1].data
ang_sat  = fits.open('/mnt/clemente/lensing/redMaPPer/angles_redMapper_forprofile.fits')[1].data

t_BG  = clusters.deVPhi_BG

mneg        = t_BG < 0.
t_BG[mneg]  = 360. + t_BG[mneg]  
t_BG        = (360. - t_BG) + 90.
mmas        = t_BG > 360.
t_BG[mmas]  = t_BG[mmas] - 360.
mmas        = t_BG > 180.
t_BG[mmas]  = t_BG[mmas] - 180.
t_BG        = np.deg2rad(t_BG)


t     = ang_sat.theta
t_wl  = ang_sat.theta_wlum
t_wd  = ang_sat.theta_wd
t_p   = ang_sat.theta_pcut
t_pwl = ang_sat.theta_pcut_wlum
t_pwd = ang_sat.theta_pcut_wd

tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='ID', format='K', array=clusters.ID),
        fits.Column(name='theta_sat_w1', format='D', array=t),
        fits.Column(name='theta_sat_wl', format='D', array=t_wl),
        fits.Column(name='theta_sat_wd', format='D', array=t_wd),
        fits.Column(name='theta_sat_pw1', format='D', array=t_p),
        fits.Column(name='theta_sat_pwl', format='D', array=t_pwl),
        fits.Column(name='theta_sat_pwd', format='D', array=t_pwd),
        fits.Column(name='theta_bcg', format='D', array=t_BG)])
        

tbhdu.writeto('/mnt/clemente/lensing/redMaPPer/redmapper_axis_angles_proxies.fits',overwrite=True)        

'''


RA0  = clusters.RA
DEC0 = clusters.DEC

LensCat.CS82.load()
ra  = np.array(LensCat.CS82.data.RAJ2000)
dec = np.array(LensCat.CS82.data.DECJ2000)
ra[ra > 275] = ra[ra>275] - 360.

# periodic = {0: (0, 360)}
# data    = np.array([ra,dec]).T
# centres = np.array([RA0,DEC0]).T
# gsp = GriSPy(data, periodic=periodic, metric='vincenty')
# near_dist, near_ind = gsp.nearest_neighbors(centres, n=1)

c            = SkyCoord(ra=RA0*u.degree, dec=DEC0*u.degree)
catalog      = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
dx, d2d, d3d = c.match_to_catalog_3d(catalog)
'''
