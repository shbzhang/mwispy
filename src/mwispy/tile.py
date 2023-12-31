' mwispy module: tile '

__version__ = '1.0'
__author__ = 'Shaobo Zhang'

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

def modifyframe(hdr, gl_ref=180):
	hdr['CRVAL1'] = gl_ref
	if hdr['CRPIX1']<0:
		hdr['CRPIX1'] += 43200
	hdr['CRPIX1'] -= gl_ref/0.0083333333333
	return hdr

def tile(files, output='tile.fits'):
	###ONLY for MOSAICKED data.
	#find range
	nfile = len(files)
	mhdr = fits.getheader(files[0])
	mhdr = modifyframe(mhdr)
	mwcs = WCS(mhdr, naxis=2)
	xran = np.ndarray((nfile,2))
	yran = np.ndarray((nfile,2))
	for i,file in enumerate(files):
		hdr = fits.getheader(file)
		hdr = modifyframe(hdr)
		wcs = WCS(hdr, naxis=2)
		lb = wcs.pixel_to_world([0, hdr['NAXIS1']-1], [0, hdr['NAXIS2']-1])
		xy = mwcs.world_to_pixel(lb)
		xran[i] = xy[0]
		yran[i] = xy[1]
	xran=np.round(xran).astype(int)
	yran=np.round(yran).astype(int)
	print(xran)

	#header
	#mhdr['NAXIS'] = 2
	mhdr['NAXIS1'] = xran.max()-xran.min()+1
	mhdr['NAXIS2'] = yran.max()-yran.min()+1
	mhdr['CRPIX1'] -= xran.min()
	mhdr['CRPIX2'] -= yran.min()
	mshp = []
	for i in range(mhdr['NAXIS'],0,-1):
		mshp.append(mhdr['NAXIS%1i' % i])
	mdat = np.zeros(mshp,dtype=np.float32)-1e3
	xran -= xran.min()
	yran -= yran.min()
	for i,afile in enumerate(files):
		dat = fits.open(afile)[0].data
		msk = np.isnan(dat)
		#msk = np.ones_like(dat, dtype=np.bool)
		#msk[16:75, 16:75] = False
		dat[msk] = 0
		mdat[...,yran[i,0]:yran[i,1]+1,xran[i,0]:xran[i,1]+1] = mdat[...,yran[i,0]:yran[i,1]+1,xran[i,0]:xran[i,1]+1]*msk + dat*(1-msk)

	mdat[mdat == -1e3] = np.nan

	mhdu = fits.PrimaryHDU(mdat, mhdr)
	mhdu.writeto(output, overwrite=True)


if __name__ == '__main__':
	import glob
	#files = glob.glob('/Users/shaobo/Work/mwips/GuideMap/*[85]*_U_m0.fits')
	files = glob.glob('/share/data/mwisp/G090+00/*U_rms.fits')
	tile(files, 'tile_U_m0.fits')
