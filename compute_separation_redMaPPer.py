import sys
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70., Om0=0.3, Ode0=0.7)

clusters = fits.open('/home/eli/Documentos/Astronomia/posdoc/halo-elongation/redMapper/redmapper_dr8_public_v6.3_catalog.fits')[1].data

RA  = clusters.RA
DEC = clusters.DEC
z   = clusters.Z_LAMBDA

Dcosmo = cosmo.comoving_distance(z)
catalog = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=Dcosmo)

idx, d2d, d3d = catalog.match_to_catalog_3d(catalog, nthneighbor=2)


mz = (z >= (z[j] - 0.02)*(z < (z[j] - 0.02)
