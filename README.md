# mwispy
``mwispy`` derives from the [Milky Way Imaging Scroll Painting (MWISP) project](https://ui.adsabs.harvard.edu/abs/2019ApJS..240....9S).

> The MWISP project is an unbiased Galactic plane CO survey for mapping region of l=-10° to +250° and |b|<=5.°2 with the 13.7 m telescope of the Purple Mountain Observatory.

The project produces tens of thousands of datacubes (cells) in FITS format, and their corresponding documents.

``mwispy`` contains a set of tools to reduce data of MWISP project.

``mwispy`` also contains the ``DataCube`` class (a subclass of [SpectralCube](https://spectral-cube.readthedocs.io/en/latest/)) to reduce datacube of the project.


## Install
Obtain the latest PyPI release with:
```sh
pip install mwispy
```

Upgrade with:
```sh
pip install --upgrade mwispy
```


## Usage
``mwispy`` package could be simply imported with:
```python
import mwispy
```

Or import all functions with:
```python
from mwispy import *
```


### Mosaic
To mosaic cells to a larger datacube:
```python
from mwispy import mosaic
mosaic(30, 40, -5, +5, -200, +200, sb='U', path='./', output='G30')
```
* _The function produces a set of mosaicked FITS files, including the datacube, noise, and coverage;_
* _See more keywords available for ``mosaic`` in its help._


### Moment
To calculate the moment of a datacube:
```python
from mwispy import cubemoment
cubemoment('datacube.fits', crange=[-15, 15], direction='v')
```
* _``crange`` is the velocity range in the unit of km/s._
* _``cubemoment`` contains a ``goodlooking`` mode which will filter the noise and make the result looking better.
* _See more keywords available for ``cubemoment`` in its help._

To derive a longitude-velocity (LV) map:
```python
from mwispy import cubemoment
cubemoment('datacube.fits', crange=[-1.5, 2.5], direction='b')
```
* _set ``direction`` to `'l'` to derive a BV map._


### PV slice
To extract position-velocity map from a datacube:

```python
from mwispy import pvslice
pvslice('datacube.fits', path_coord=[[80,81,82], [-1,1,2]], width=5, step=0.5)
```
* _The function produces three files: the slice map, the path, and the outline of belt._
* _See more keywords available for ``pvslice`` in its help._


### Tile
To tile 2-d images derived from separately mosaicked datacube.
```python
from mwispy import tile
from glob import glob
tile(glob('*_U_m0.fits'), output='tile_m0.fits')
tile(glob('*_U_lvmap.fits'), output='tile_lvmap.fits')
```


### Other utilities
To convert velocity to/from channel:
```python
from mwispy import v2c, c2v
from astropy.io import fits
hdr = fits.getheader('datacube.fits')
chan = v2c(hdr, [-10, 10])
vaxis = c2v(hdr, range(hdr['NAXIS']))
```

To convert a cell name to/from the coordiante:
```python
from mwispy import cell2lb, lb2cell
l, b = cell2lb('0345+015')
cell = lb2cell(31.5, -3.5)
```


## DataCube class
The ``DataCube`` class provides an alternative way to analyze datacube. 

``DataCube`` inherits most methods from the ``SpectralCube`` class, and adds several new methods to analyze and visualize datacubes. Analysis like moment could also be done with these methods, but using the above funcions is more efficient.

### Open a MWISP datacube
```python
from mwispy import DataCube
from astropy.coordinates import SkyCoord
import astropy.units as u

cube = DataCube.openMWISP('datacube.fits')
```


### Moment
This could be done using inherited methods
```python
#moment 0 map
subcube = cube.spectral_slab(-15*u.km/u.s, 15*u.km/u.s)
m0 = subcube.moment(order=0)
m0.write('datacube_m0.fits')

#LV map
subcube = cube.subcube(ylo=-1.5*u.deg, yhi=1.5*u.deg)
lv = subcube.moment(order=0, axis=1)
lv.write('datacube_lvmap.fits')
```

### Append a RMS image
The rms image will be used for average spectra, or signal detection.
```python
cube = cube.with_rms('datacube_rms.fits')
#access the rms
cube.rms
```

### Masking along axis
Mask voxel in the datacube above a value or user defined function.

Especially useful when calculating some properties with different velocity range at different spatial position.
```python
#masking along the l axis
llo = lambda b,v: b+86*u.deg
mask = cube.x_mask(llo)
maskedCube = cube.with_mask(mask)

#masking along the b axis
blo = lambda l,v: l-86*u.deg
mask = cube.y_mask(blo)
maskedCube = cube.with_mask(mask)

#masking along the v axis
vflo = lambda l,b: ((l/u.deg-84.5)**2*1.5-12.5)*u.km/u.s
vfhi = 10*u.km/u.s
mask = cube.z_mask(vflo) & ~cube.z_mask(vfhi)
maskedCube = cube.with_mask(mask)

#masking noise
cube = cube.with_rms('datacube_rms.fits')
mask = cube.v_mask(userms=True, threshold=3, consecutive=3)
peakv = cube.with_mask(mask)
```

### Get velocity
Similar as function ``c2v``.
```python
#get velocity axis of a DataCube
vaxis = cube.velocity()
#Convert channel to velcity
velocity = cube.velocity(channel = [30, 100, 220], vunit = 'km/s')
```

### Get channel
Similar as function ``v2c``.
```python
#get 0-base channel axis of a DataCube
caxis = cube.channel()
#Convert velcity to channel
channel = cube.channel(velocity = [-15, 5, 20], vunit = 'km/s')
```

### PV slice
Similar as function ``pvslice``. The method returns a HDUList contian the slice map, and the path.
```python
path = SkyCoord([80,81,82], [-1,1,2], frame='galactic', unit='deg')
pvhdu = cube.pvslice(path, width=5, step=0.5)
pvhdu.write('pvslice.fits')
```

### Baseline fitting
```python
#set mode and window
fittedCube = cube.with_window(-60, -40, -20, 20, modex = [-200, 200], vunit='km/s')
#do the baseline fitting
fittedCube.baseline(deg=1)
#calculate new RMS
rms = fittedCube.rms()
```

### Average spectra
```python
#average with equal weighting
avsp1 = cube.average()
#average with noise weighting
avsp2 = cube.with_rms('datacube_rms.fits').average()
#plot spectra
import matplotlib.pyplot as plt
plt.step(cube.velocity(), avsp1._data, label='Equal weighting')
plt.step(cube.velocity(), avsp2._data, label='RMS weighting')
plt.show()
```

### Rebin velocity
Smooth and rebin the velocity.
```python
#use rebinvelocity
rebinnedCube = cube.rebinvelocity(-10, 10, 21, vunit='km/s', largecube=True)

#use resample to align with another header
ResampleDataCube = cube.resample(referenceheader, vunit='km/s')
#use resample with values (NumC, RefC, RefV, IncV)
ResampleDataCube = cube.resample(201, 100, 0.0, 1.0, vunit='km/s')
```

### Plot channel map
```python
#rebin velocity first
cube = cube.rebinvelocity(-10, 10, 21, vunit='km/s', largecube=True)

#plot
fig, ax = cube.channelmap(nrows=3, ncols=7, vunit='km/s', figureheight=8, \
	imshow_kws = dict(vmin=-0.1, vmax=5, cmap='rainbow'),\
	contour_kws = dict(levels=[1,2,3,4,5,6]),\
	text_kws = dict(x=0.1, y=0.1, size=5),\
	tick_kws = dict(size=10, direction='in'),\
	colorbar_kws = None)

#ax[-1,0] is the lower left pannel with tick labels
ax[-1,0].set_xlabel('glon')
ax[-1,0].set_ylabel('glat')
ax[-1,0].coords[0].set_major_formatter('d.d')
ax[-1,0].coords[1].set_major_formatter('d.d')
plt.show()
```

### Plot grid spectra
Plot spectra in grid like ``Gildas``
```python
#extract required region
cube = cube[:, 10:20, 15:25]
fig, ax = cube.gridmap()
plt.show()
```

### Plot peakvelocity map
Each pixel in the map represents the velocity of the spectral peak along the line of sight.
Masking noise before doing this shows better result.
```python
#masking noise
cube = cube.with_rms('datacube_rms.fits')
mask = cube.v_mask(userms=True, threshold=3, consecutive=3)
peakv = cube.with_mask(mask).peakvelocity()

#plot the map
ax = plt.subplot(projection=peakv.wcs)
ax.imshow(peakv.data)
plt.show()
```