' mwispy module: converter '

__version__ = '1.0'
__author__ = 'Shaobo Zhang'
__all__ = ['c2v', 'v2c', 'lb2cell', 'cell2lb']

import re
import numpy as np

def c2v(header, channel):
	'''
	Convert channel to velocity using header

	Parameters
	----------
	header : astropy.io.fits header object
		FITS header based on.
	channel : array_like
		0-based channels to be converted.

	returns
	-------
	velocity : ndarray with the same shape as velocity
		converted velocity.

	Examples
	--------
	>>> header = fits.getheader('0340+010U.fits')
	>>> c = [0,9000,9023]
	>>> v = mwisp.c2v(header, c)
	>>> v   #in m/s
	array([-1428638.8009788 ,        0.        ,     3650.96582472])
	'''
	channel = np.array(channel)
	nc = header['NAXIS3']
	v0 = header['CRVAL3']
	c0 = header['CRPIX3']
	dv = header['CDELT3']
	velocity = (channel-c0+1)*dv+v0
	return velocity


def v2c(header, velocity):
	'''
	Convert velocity to channel using header

	Parameters
	----------
	header : astropy.io.fits header object
		FITS header based on.
	velocity : array_like
		velocity to be converted.

	returns
	-------
	channel : ndarray with the same shape as velocity
		converted 0-based channels.

	Examples
	--------
	>>> header = fits.getheader('0340+010U.fits')
	>>> v = [-10000, 0, 20000]   #in m/s
	>>> c = mwisp.v2c(header, v)
	>>> c
	array([ 8937.00297098,  9000.        ,  9125.99405803])
	'''
	velocity = np.array(velocity)
	nc = header['NAXIS3']
	v0 = header['CRVAL3']
	c0 = header['CRPIX3']
	dv = header['CDELT3']
	channel = (velocity-v0)/dv+c0-1
	return channel


def lb2cell(gl, gb):
	'''
	Convert gl, gb to cell name

	Parameters
	----------
	gl : float
		galactic longitude in degree.
	gb : float
		galactic latitude in degree.

	returns
	-------
	cellname : str
		name of the MWISP cell that contains (gl, gb).

	Examples
	--------
	>>> cell = mwisp.lb2cell(102.1, -2.6)
	>>> cell
	'1020-025'
	'''
	gl = round(gl*2)*5
	gb = round(gb*2)*5
	gl = gl % 3600
	return str('%04d%+04d' % (gl, gb))


def cell2lb(cellname):
	'''
	Convert cell name to gl, gb

	Parameters
	----------
	cellname : str
		name of the MWISP cell that contains (gl, gb).

	returns
	-------
	gl : float
		galactic longitude in degree.
	gb : float
		galactic latitude in degree.

	Examples
	--------
	>>> gl, gb = mwisp.cell2lb('0835-025')
	>>> gl, gb
	(83.5, -2.5)
	'''
	reg = r'[0123]\d\d[05][\+\-]\d\d[05]'
	if re.match(reg, cellname): return float(cellname[0:4])/10, float(cellname[4:8])/10
	else: raise ValueError('faulty cellname: %s' % cellname)
