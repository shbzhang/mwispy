import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from easyplotlib import EasyNorm
import io


mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2

class LinearWCS():
	def __init__(self, header, axis):
		self.wcs =  {key:header.get('%5s%1i' % (key, axis)) for key in ['NAXIS','CTYPE','CRVAL','CDELT','CRPIX','CROTA']}
		self.longitude = axis==1

	def pixel_to_world(self, pixel, base=0):
		world = (np.array(pixel)-self.wcs['CRPIX']+1-base)*self.wcs['CDELT']+self.wcs['CRVAL']
		if self.longitude: world = world % 360
		return world

	def world_to_pixel(self, world, base=0):
		pixel = (np.array(world)-self.wcs['CRVAL'])/self.wcs['CDELT']+self.wcs['CRPIX']-1+base
		if self.longitude: pixel = pixel % abs(360//self.wcs['CDELT'])
		return pixel

	@property
	def axis(self):
		return self.pixel_to_world(np.arange(self.wcs['NAXIS']))

	@property
	def extent(self):
		#imshow extent
		return self.pixel_to_world([-0.5, self.wcs['NAXIS']-0.5])


def resample_lvmap(file, ref, output):
	hdu = fits.open(file)[0]
	wcs = LinearWCS(hdu.header, 2)
	refhdu = fits.open(ref)[0]
	refwcs = LinearWCS(refhdu.header, 2)

	v = wcs.axis
	refv = refwcs.axis
	refx = np.arange(refhdu.header['NAXIS1'])

	refnan = np.isnan(refhdu.data)

	hdu.data[np.isnan(hdu.data)] = 0
	interpfunc = RectBivariateSpline(v, refx, hdu.data)
	resample = interpfunc(refv, refx)
	resample[refnan] = np.nan
	refhdu.data=resample
	refhdu.writeto(output, overwrite=True)
	return output


class ShowTiles():
	def __init__(self, \
		parts=1, \
		lvmap=False, \
		rmsmap = False, \
		lrange=(9.75, 229.75), \
		#lrange=(028.8012-74/60/np.cos(+03.4978/180*np.pi), 028.8012+60/60/np.cos(+03.4978/180*np.pi)), \
		#lrange=(25,50), \
		#brange=(+03.4978-46/60, +03.4978+86/60), \
		brange=(-5.25,5.25), \
		vrange=(-200.0, 250.0), \
		loverlap = 2, \
		zero = 0., \
		axis_off = False, \
		quality = 'med', \
		suffix = 'tile', \
		exportformat = 'png', \
		):

		self.parts = parts
		self.lvmap = lvmap
		self.rmsmap = rmsmap
		self.lrange = (min(lrange), max(lrange))
		self.brange = (min(brange), max(brange))
		self.vrange = (min(vrange), max(vrange))
		self.loverlap = loverlap # overlap of seprate l panels
		self.zero = zero # plot zeros as 0=black, 1=white, 0.X gray
		self.axis_off = axis_off
		self.quality = quality
		self.suffix = suffix
		self.figure = None
		self.exportformat = exportformat

	@property
	def lspan(self):
		return self.lrange[1] - self.lrange[0]
	@property
	def bspan(self):
		return self.brange[1] - self.brange[0]
	@property
	def vspan(self):
		return self.vrange[1] - self.vrange[0]
	@property
	def lwidth(self):
		return (self.lspan + self.loverlap * (self.parts-1)) / self.parts
	@property
	def xlim(self):
		width = self.lwidth
		xlim = []
		for i in range(self.parts):
			right = self.lrange[0] + (width-self.loverlap)*i
			xlim.append((right+width, right))
		#sep = np.linspace(self.lrange[0], self.lrange[1]+self.loverlap*(self.parts), self.parts+1)
		return xlim#[(sep[i+1], sep[i]) for i in range(self.parts)]
	@property
	def ylim(self):
		if self.lvmap: return self.vrange
		else: return self.brange
	@property
	def unit(self):
		if self.lvmap: return 'K deg'
		elif self.rmsmap: return 'K'
		else: return 'K km s$^{-1}$'


	def _getFigure(ncols, nrows, width, height, left, right, bottom, top, wspace, hspace, dpi):
		figsize = ( \
			left + width*ncols + wspace*(ncols-1) + right, \
			bottom + height*nrows + hspace*(nrows-1) + top)

		fig = plt.figure(figsize = figsize, dpi = dpi)
		fig.subplots_adjust(left = left / figsize[0], right = 1 - right / figsize[0], \
			bottom = bottom / figsize[1], top = 1 - top / figsize[1], \
			wspace = wspace / width, hspace = hspace / height)
		return fig


	def createFigure(self, \
		separate=False, \
		degreePerInch=5, \
		lvratio=10, \
		fontsize=12, \
		marginLeft = 5, \
		marginRight = 3, \
		marginTop = 2, \
		marginBottom = 3, \
		widthSpace = 3, \
		heightSpace = 2.5, \
		):

		marginLeft = 0 if self.axis_off else marginLeft
		marginRight = 0 if self.axis_off else marginRight
		marginTop = 0 if self.axis_off else marginTop
		marginBottom = 0 if self.axis_off else marginBottom
		widthSpace = 0 if self.axis_off else widthSpace
		heightSpace = 0 if self.axis_off else heightSpace

		'''
		degreePerInch = deg/inch
		margin in degree
		'''
		self.separate = separate
		self.degreePerInch = degreePerInch
		self.lvratio = lvratio
		self.fontsize = fontsize

		###ax size, keep II and LV have the same width
		axesWidth = self.lwidth / degreePerInch
		axesHeight = self.bspan / degreePerInch
		if self.lvmap:
			self.axesSize = (axesWidth, axesHeight * self.lvratio)
		else:
			self.axesSize = (axesWidth, axesHeight)

		###dpi
		real_dpi = round(degreePerInch * 60 * 2)
		DPI = {'low':real_dpi//4, 'med':real_dpi//2, 'high': real_dpi*2}
		if self.quality in DPI.keys():
			self.dpi = DPI[self.quality]
		else: raise ValueError("quality must be '%s', '%s', or '%s'" %  DPI.keys())

		###figsize
		self.marginLeft = marginLeft / degreePerInch	#convert from degree to inch
		self.marginRight = marginRight / degreePerInch
		self.marginBottom = marginBottom / degreePerInch
		self.marginTop = marginTop / degreePerInch
		self.widthSpace = widthSpace / degreePerInch
		self.heightSpace = heightSpace / degreePerInch
		if separate:
			self.figsize = (self.axesSize[0] + self.marginLeft + self.marginRight, \
				self.axesSize[1] + self.marginBottom + self.marginTop)
		else:
			self.figsize = (self.axesSize[0] + self.marginLeft + self.marginRight, \
				self.axesSize[1] * self.parts + self.heightSpace * (self.parts-1) + self.marginBottom + self.marginTop)

		if separate:
			self.figure = [ShowTiles._getFigure(1, 1, *self.axesSize, \
				self.marginLeft, self.marginRight, self.marginBottom, self.marginTop, \
				self.widthSpace, self.heightSpace, self.dpi) for i in range(self.parts)]
		else:
			self.figure = ShowTiles._getFigure(1, self.parts, *self.axesSize, \
				self.marginLeft, self.marginRight, self.marginBottom, self.marginTop, \
				self.widthSpace, self.heightSpace, self.dpi)

		print('Plot %s map %s.' % ('Glon-Velo' if self.lvmap else 'IntIntensity', 'separately' if self.separate else 'together'))


	def _readTable(self, file):
		f = open(file, 'r')
		l = []
		b = []
		v = []
		sl = []
		sb = []
		sv = []
		s = []
		for line in f:
			if line[0] == '#': continue
			line = line.split()
			if (float(line[0])<self.lrange[0]) | (float(line[0])>self.lrange[1]): continue
			if (float(line[1])<self.brange[0]) | (float(line[1])>self.brange[1]): continue
			l.append(float(line[0]))
			b.append(float(line[1]))
			v.append(float(line[2]))
			sl.append(float(line[3]))
			sb.append(float(line[4]))
			sv.append(float(line[5]))
			s.append(' '.join(line[6:]))
		return l, b, v, sl, sb, sv, s


	def adjustLabel(self, ax, label):
		from adjustText import adjust_text

		masky, maskx = np.argwhere(self.mask > 0.3).T
		index = (maskx%8==0) | (masky%8==0)
		maskl = list(LinearWCS(self.header, 1).pixel_to_world(maskx[index]))
		maskb = list(LinearWCS(self.header, 2).pixel_to_world(masky[index]))
		ax.plot(maskl, maskb, 'w.', markersize=2)
		#print(len(maskl))

		f = open(label, 'r')
		l, b, v, _, _, _, s = self._readTable(label)
		t = [ax.text(l[i], b[i], s[i], rotation=0, fontsize=self.fontsize*0.65, fontweight='bold', clip_on=True) for i in range(len(l))]

		ax.plot(l, b, 'r.', markersize=2)
		adjust_text(t, maskl+l, maskb+b, arrowprops=dict(arrowstyle='-', color='k'))


	def drawLabel(self, ax, label, line=False):
		f = open(label, 'r')
		if self.lvmap:
			x, _, y, sx, _, sy, s = self._readTable(label)
		else:
			x, y, _, sx, sy, _, s = self._readTable(label)

		#print(self.cmap)
		firstColor = self.cmap(0)[:3]#(self.cmap._segmentdata['red'], self.cmap._segmentdata['green'], self.cmap._segmentdata['blue'])
		#print(firstColor, self.cmap.colors)
		c = (0.2,0.2,0.2) if sum(firstColor)/3 > 0.5 else (0.9,0.9,0.9)
		for i in range(len(x)):
			text = ax.text(sx[i], sy[i], s[i], rotation=0, color=c, fontsize=self.fontsize*0.65, fontweight='bold', ha='center', va='center', clip_on=True)
			if line:
				linex = np.linspace(sx[i], x[i], 300)
				liney = np.linspace(sy[i], y[i], 300)
				hide = (np.abs(linex - sx[i]) < 0.21*len(s[i])) & (np.abs(liney - sy[i]) < (6 if self.lvmap else 0.31))
				ax.plot(linex[~hide], liney[~hide], '-', color=c, linewidth=0.5)
			#ax.plot(np.array([-.2, .2, .2, -.2, -.2])*len(s[i])+sx[i], np.array([-.3,-.3,.3,.3,-.3])+sy[i], '-', color='b')
		#ax.plot(x, y, 'r.', markersize=5)


	def drawCat(self, ax, **cat_kws):
		catL = cat_kws.pop('l')
		catB = cat_kws.pop('b')
		catV = cat_kws.pop('v')
		catTag = cat_kws.pop('tag')
		catSize = cat_kws.pop('size', None)
		catColor = cat_kws.pop('color', None)
		catMarker = cat_kws.pop('marker', None)
		catLabel = cat_kws.pop('label', None)

		tags = np.unique(catTag)
		if catSize is None: catSize = {t:1 for t in tags}
		if catMarker is None: catMarker = {t:'o' for t in tags}

		#print(catL, catB, catTag)
		for t in tags:
			idx = catTag == t
			if self.lvmap:
				if isinstance(catV.iloc[0], list):
					# LHW
					for i, (l,v) in enumerate(zip(catL[idx], catV[idx])):
						ax.plot([l,l], v, '-', c=catColor[t], label=catLabel[t] if i==0 else None, **cat_kws)
				else:
					# general cat
					ax.scatter(catL[idx], catV[idx], s=catSize[t], c=catColor[t], marker=catMarker[t], label=catLabel[t], **cat_kws)
			else:
				ax.scatter(catL[idx], catB[idx], s=catSize[t], c=catColor[t], marker=catMarker[t], label=catLabel[t], **cat_kws)
		#ax.legend(loc='upper left', fontsize=self.fontsize)



	def configAxes(self, ax, index, bottom_panel=True):
		ax.set_xticks(range(0, 360, 10))
		ax.set_xticks(range(0, 360, 1), minor=True)
		if not self.axis_off:
			ax.set_xticklabels(['$%i^\circ$' % t for t in range(0, 360, 10)])
			if bottom_panel: ax.set_xlabel('Galactic Longitude', fontsize=self.fontsize*1.2)
			if self.lvmap:
				ax.set_aspect('auto')#self.bspan / self.vspan * self.lvratio)
				ax.set_ylabel('LSR radial velocity (km s$^{-1}$)', fontsize=self.fontsize*1.2)
				ax.set_yticks(range(-300, 300, 50))
				ax.set_yticks(range(-300, 300, 10), minor=True)
				ax.set_yticklabels(['$%+i$' % t if t!=0 else '$0$' for t in range(-300, 300, 50)])
			else:
				ax.set_aspect('equal')
				ax.set_ylabel('Galactic Latitude', fontsize=self.fontsize*1.2)
				ax.set_yticks(range(-10, 11, 2))
				ax.set_yticks(range(-10, 11, 1), minor=True)
				ax.set_yticklabels(['$%+i^\circ$' % t if t!=0 else '$0^\circ$' for t in range(-10, 11, 2)])
			ax.tick_params(which='both', top=True, bottom=True, labeltop=False, labelbottom=True, direction='out')
			ax.tick_params(which='both', left=True, right=True, labelleft=True, labelright=True, direction='out')
			ax.tick_params(axis='both', labelsize=self.fontsize, pad=2)
			ax.tick_params(which='major', length=5, width=1.5)
			ax.tick_params(which='minor', length=3, width=1)
		else:
			ax.set_axis_off()

		ax.set_xlim(self.xlim[index])
		ax.set_ylim(self.ylim)


	def set_ticks(norm, cbar):
		###not working yet
		if isinstance(norm, EasyNorm.SqrtNorm):
			if norm.vmax>100: cbar.set_ticks(np.arange(0, norm.vmax*1.01, 2)**2)
			elif norm>25: cbar.set_ticks(np.arange(0, norm.vmax*1.01, 1)**2)
			elif norm>6.25: cbar.set_ticks(np.arange(0, norm.vmax*1.01, 0.5)**2)


	def colorbar(self):
		if len(self.norm) == 0: return
		elif len(self.norm) == 1:
			mappable = mpl.cm.ScalarMappable(cmap=self.cmap, norm=self.norm[0])

			###vertical colorbar
			fig = ShowTiles._getFigure(1, 1, self.axesSize[1]*0.05, self.axesSize[1], \
				self.marginLeft*0.02, self.marginRight, self.marginBottom, self.marginTop, \
				self.widthSpace, self.heightSpace, self.dpi)
			cb = fig.colorbar(mappable, cax=fig.add_subplot(), orientation = 'vertical')
			cb.ax.tick_params(labelsize=self.fontsize)
			cb.ax.set_title(self.unit, loc='left', fontsize = self.fontsize*0.8)
			#cb.ax.minorticks_on()
			output = 'S%i_0_%s_colorbarV.%s' % (self.parts, self.suffix, self.exportformat)
			fig.savefig(output, format=self.exportformat)
			plt.close(fig)

			###horizental colorbar
			fig = ShowTiles._getFigure(1, 1, self.axesSize[0], self.axesSize[0]*0.01, \
				self.marginLeft, self.marginRight, self.marginBottom, self.marginTop*0.02, \
				self.widthSpace, self.heightSpace, self.dpi)
			cb = fig.colorbar(mappable, cax=fig.add_subplot(), orientation = 'horizontal')
			cb.ax.tick_params(labelsize=self.fontsize)
			if self.parts==1:
				cb.ax.text(1, -0.95, self.unit, transform=cb.ax.transAxes, fontsize = self.fontsize*0.8, ha='center', va='top')
			elif self.parts==3:
				cb.ax.text(1, -1.7, self.unit, transform=cb.ax.transAxes, fontsize = self.fontsize*0.8, ha='center', va='top')
			output = 'S%i_0_%s_colorbarH.%s' % (self.parts, self.suffix, self.exportformat)
			fig.savefig(output, format=self.exportformat)
			plt.close(fig)

		else:
			###vertical colorbar
			fig = ShowTiles._getFigure(1, 3, self.axesSize[1]*0.05, self.axesSize[1], \
				self.marginLeft*0.02, self.marginRight, self.marginBottom, self.marginTop, \
				self.widthSpace, self.heightSpace, self.dpi)
			for i in range(3):
				cmap = mpl.colors.ListedColormap(np.linspace(0,1,256)[:,np.newaxis]*np.array([i==0,i==1,i==2]))
				mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=self.norm[i])
				cb = fig.colorbar(mappable, cax=fig.add_subplot(3, 1, 3-i), orientation = 'vertical')
				cb.ax.tick_params(labelsize=self.fontsize)
			cb.ax.set_title(self.unit, loc='left', fontsize = self.fontsize*0.8)
			output = 'S%i_0_%s_colorbarV.%s' % (self.parts, self.suffix, self.exportformat)
			fig.savefig(output, format=self.exportformat)
			plt.close(fig)

			###horizental colorbar
			fig = ShowTiles._getFigure(3, 1, (self.axesSize[0]-self.widthSpace*2)/3, (self.axesSize[0]-self.widthSpace*2)/3*0.01, \
				self.marginLeft, self.marginRight, self.marginBottom, self.marginTop*0.2, \
				self.widthSpace, self.heightSpace, self.dpi)
			for i in range(3):
				cmap = mpl.colors.ListedColormap(np.linspace(0,1,256)[:,np.newaxis]*np.array([i==0,i==1,i==2]))
				mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=self.norm[i])
				cb = fig.colorbar(mappable, cax=fig.add_subplot(1, 3, i+1), orientation = 'horizontal')
				cb.ax.tick_params(labelsize=self.fontsize)
			if self.parts==1:
				cb.ax.text(1, -1.8, self.unit, transform=cb.ax.transAxes, fontsize = self.fontsize*0.8, ha='center', va='top')
			elif self.parts==3:
				cb.ax.text(1, -6.0, self.unit, transform=cb.ax.transAxes, fontsize = self.fontsize*0.8, ha='center', va='top')
			output = 'S%i_0_%s_colorbarH.%s' % (self.parts, self.suffix, self.exportformat)
			fig.savefig(output, format=self.exportformat)
			plt.close(fig)


	def exportFullRes(self, header, data, label, all_zeros=0, rebin=2):
		#all_zeros: alpha 0=transparent, 1=black
		from PIL import Image, ImageCms
		dataHeight = data.shape[0]
		lOverlapPixel = round(self.loverlap // abs(header['CDELT1']))
		# find a closest overlap that each panel has the same width in pixel
		for lop in range(lOverlapPixel, 0, -1):
			dataWidth = (data.shape[1] + lop*(self.parts-1))/self.parts
			if dataWidth - int(dataWidth) < 0.001:
				lOverlapPixel = lop
				dataWidth = round(dataWidth)
				break

		#print(data.shape, 'overlap:', lOverlapPixel, 'axes w:', dataWidth, 'axes h:', dataHeight)
		### size in pixel
		marginLeft = 600
		marginRight = 400
		marginTop = 200
		marginBottom = 400
		heightSpace = 290
		widthSpace = 250

		spineWidth = 11
		tickWidth = 11
		tickMajor = 40
		tickMinor = 15
		ticklabelFontsize = 25
		ticklabelPadding = 70
		labelFontsize = int(ticklabelFontsize*1.2)
		labelPadding = 80

		textFontsize = int(ticklabelFontsize*0.65)

		axesWidth = dataWidth + spineWidth*2
		axesHeight = dataHeight + spineWidth*2
		width = marginLeft + axesWidth + marginRight
		height = marginTop + axesHeight * self.parts + heightSpace * (self.parts-1) + marginBottom
		#print('w',width,'h',height,'aw',axesWidth,'ah',axesHeight)

		lwcs = LinearWCS(header, 1)
		bwcs = LinearWCS(header, 2)

		lminor = lwcs.world_to_pixel(np.arange(0, 240, 1)).astype(int)
		lmajor = lwcs.world_to_pixel(np.arange(0, 240, 10)).astype(int)
		lticklabels = np.array(['%i$^\circ$' % l for l in np.arange(0, 240, 10)])

		bminor = bwcs.world_to_pixel(np.arange(-5, 6, 1)).astype(int)
		bmajor = bwcs.world_to_pixel(np.arange(-6, 6, 2)).astype(int)
		bticklabels = np.array(['$0^\circ$' if b==0 else '$%+i^\circ$' % b for b in np.arange(-6, 6, 2)])


		def text2pixel(text, **kws):
			###Render a single line of text to a binary or alpha mask.
			# draw text
			dpi=100
			fig, ax = plt.subplots(figsize=(1,1), dpi=dpi)
			fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
			ax.set_axis_off()
			text = ax.text(0.5, 0.5, text, color='black', ha='center', va='center', **kws)
			# write to buffer
			buf = io.BytesIO()
			fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
			plt.close(fig)
			buf.seek(0)
			# read mask, clip margin
			im = Image.open(buf)
			mask = np.array(im)[...,3]
			while (mask[0]==0).all(): mask = mask[1:]
			while (mask[-1]==0).all():mask = mask[:-1]
			while (mask[:,0]==0).all():mask = mask[:,1:]
			while (mask[:,-1]==0).all():mask = mask[:,:-1]
			return mask/255.


		### arr init
		imarr = np.empty((height, width, 4), dtype=np.uint8)
		imarr[:] = 255

		maxXTicklabelHeight = 0
		maxYTicklabelWidth = 0
		### panels
		for i in range(self.parts):
			dataOffset = (dataWidth-lOverlapPixel)*(self.parts-i-1)
			### spine
			x0 = marginLeft
			x1 = marginLeft + axesWidth
			y0 = marginTop+heightSpace*i+axesHeight*i
			y1 = marginTop+heightSpace*i+axesHeight*(i+1)
			imarr[y0:y1, x0:x1, :3] = 0

			
			### data
			xim0 = marginLeft + spineWidth
			xim1 = marginLeft + axesWidth - spineWidth
			yim0 = marginTop + heightSpace*i + axesHeight*i + spineWidth
			yim1 = marginTop + heightSpace*i + axesHeight*(i+1) - spineWidth
			imarr[yim0:yim1, xim0:xim1, :3] = (data[::-1][:, dataOffset:dataOffset+dataWidth, :] * 255).astype(np.uint8)
			### zero
			zeros = (data[::-1][:, dataOffset:dataOffset+dataWidth, :] == 0).all(axis=2)
			imarr[yim0:yim1, xim0:xim1, 3][zeros] = int(all_zeros*255)


			### label
			if label is not None:
				_, _, _, sx, sy, _, s = self._readTable(label)
				for m in range(len(sx)):
					ta = text2pixel(s[m], fontsize=textFontsize, fontweight='bold')
					xt = marginLeft + spineWidth + round(lwcs.world_to_pixel(sx[m])) - dataOffset - ta.shape[1]//2
					yt = marginTop + heightSpace*i + axesHeight*(i+1) - spineWidth - round(bwcs.world_to_pixel(sy[m])) - ta.shape[0]//2
					#print(i, xt, yt, s[m])
					if xt<xim0 or xt+ta.shape[1]>=xim1: continue
					if yt<yim0 or yt+ta.shape[0]>=yim1: continue
					###img
					imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3] = \
						(imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3]*(1-ta[..., None]) + 
						np.array([230, 230, 230], dtype=np.uint8) * ta[..., None]).astype(np.uint8)
					###alpha
					imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], 3] = \
						(imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], 3]*(1-ta) + 
						255 * ta).astype(np.uint8)
			

			### xticks
			# major, ticklabels
			xmajorInside = (lmajor >= dataOffset) & (lmajor < dataOffset+dataWidth)
			for x, lt in zip(lmajor[xmajorInside], lticklabels[xmajorInside]):
				# tick
				xt0 = x0+spineWidth + x-dataOffset - tickWidth//2
				xt1 = x0+spineWidth + x-dataOffset + tickWidth//2+1
				yt0 = y0-tickMajor
				yt1 = y1+tickMajor
				imarr[yt0:y0, xt0:xt1, :3] = 0
				imarr[y1:yt1, xt0:xt1, :3] = 0
				# text
				ta = text2pixel(lt, fontsize=ticklabelFontsize)
				xt = x0+spineWidth + x-dataOffset - ta.shape[1]//2
				yt = y1+ticklabelPadding# + ta.shape[0]//2
				imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3] = \
					(imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3]*(1-ta[..., None]) + 
					np.array([0, 0, 0], dtype=np.uint8) * ta[..., None]).astype(np.uint8)
				if ta.shape[0] > maxXTicklabelHeight: maxXTicklabelHeight = ta.shape[0]
			# minor
			xminorInside = (lminor >= dataOffset) & (lminor < dataOffset+dataWidth)
			for x in lminor[xminorInside]:
				xt0 = x0+spineWidth + x-dataOffset - tickWidth//2
				xt1 = x0+spineWidth + x-dataOffset + tickWidth//2+1
				yt0 = y0-tickMinor
				yt1 = y1+tickMinor
				imarr[yt0:y0, xt0:xt1, :3] = 0
				imarr[y1:yt1, xt0:xt1, :3] = 0

		
			### yticks
			# major, ticklabels
			ymajorInside = (bmajor >= 0) & (bmajor < dataHeight)
			for y, bt in zip(bmajor[ymajorInside], bticklabels[ymajorInside][::-1]):
				# tick
				xt0 = x0-tickMajor
				xt1 = x1+tickMajor
				yt0 = y0+spineWidth + y - tickWidth//2
				yt1 = y0+spineWidth + y + tickWidth//2+1
				imarr[yt0:yt1, xt0:x0, :3] = 0
				imarr[yt0:yt1, x1:xt1, :3] = 0
				# text
				ta = text2pixel(bt, fontsize=ticklabelFontsize)
				xt = x0-ticklabelPadding
				yt = y0+spineWidth + y - ta.shape[0]//2
				imarr[yt:yt+ta.shape[0], xt-ta.shape[1]:xt, :3] = \
					(imarr[yt:yt+ta.shape[0], xt-ta.shape[1]:xt, :3]*(1-ta[..., None]) + 
					np.array([0, 0, 0], dtype=np.uint8) * ta[..., None]).astype(np.uint8)
				xt = x1+ticklabelPadding
				imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3] = \
					(imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3]*(1-ta[..., None]) + 
					np.array([0, 0, 0], dtype=np.uint8) * ta[..., None]).astype(np.uint8)
				if ta.shape[1] > maxYTicklabelWidth: maxYTicklabelWidth = ta.shape[1]
			# minor
			yminorInside = (bminor >= 0) & (bminor < dataHeight)
			for y in bminor[yminorInside]:
				xt0 = x0-tickMinor
				xt1 = x1+tickMinor
				yt0 = y0+spineWidth + y - tickWidth//2
				yt1 = y0+spineWidth + y + tickWidth//2+1
				imarr[yt0:yt1, xt0:x0, :3] = 0
				imarr[yt0:yt1, x1:xt1, :3] = 0


			#ylabel			
			ta = text2pixel('Galactic Latitude', fontsize=labelFontsize, rotation=90)
			xt = x0 - ticklabelPadding - maxYTicklabelWidth -labelPadding
			yt = y0+axesHeight//2 - ta.shape[0]//2
			imarr[yt:yt+ta.shape[0], xt-ta.shape[1]:xt, :3] = \
				(imarr[yt:yt+ta.shape[0], xt-ta.shape[1]:xt, :3]*(1-ta[..., None]) + 
				np.array([0, 0, 0], dtype=np.uint8) * ta[..., None]).astype(np.uint8)


		ta = text2pixel('Galactic Longitude', fontsize=labelFontsize)
		xt = x0 + axesWidth//2 - ta.shape[1]//2
		yt = y1 + ticklabelPadding + maxXTicklabelHeight + labelPadding
		imarr[yt:yt+ta.shape[0], xt:xt+ta.shape[1], :3] = \
			(imarr[yt:yt+ta.shape[0], xt-ta.shape[1]:xt, :3]*(1-ta[..., None]) + 
			np.array([0, 0, 0], dtype=np.uint8) * ta[..., None]).astype(np.uint8)
		
		output = 'S%i_%s.png' % (self.parts, self.suffix)
		#Image.fromarray(imarr).save('test.png')
		plt.imsave(output, imarr)



	def draw(self, *data, header=None, scale='sqrt', vmin=None, vmax=None, p=None, \
		label=None, adjust=False, figure_kws={}, full_res=False, cat_kws=None, all_zeros=0, **kws):
		###deal with data
		if len(data) == 0:
			self.norm = []
			img = None
			img = np.load('image.npy')
		elif len(data) == 1:
			self.norm = [EasyNorm.get(scale, vmin=vmin, vmax=vmax, p=p), ]
			img = self.norm[0](np.squeeze(data))
			img[img<0] = 0
			img[img>1] = 1
			if (label is not None) and adjust: self.mask = img
		else:
			n = len(data)
			if isinstance(scale, str): scale = [scale]*n
			if vmin is None or isinstance(vmin, (int,float)): vmin = [vmin]*n
			if vmax is None or isinstance(vmax, (int,float)): vmax = [vmax]*n
			if p is None or isinstance(p, (int,float)): p = [p]*n
			self.norm = []
			img = []

			#print(np.array(data).shape)
			zeros = np.sum(np.array(data)==0, axis=0)==3
			#print(zeros.sum())
			for i in range(n):
				normi = EasyNorm.get(scale[i], vmin=vmin[i], vmax=vmax[i], p=p[i])
				datai = np.squeeze(data[i])
				imgi = normi(datai)
				self.norm.append(normi)
				imgi[zeros] = self.zero
				img.append(imgi)

			#img.append(imgi>0.01)
			for im in img:
				print(im.shape)
			img = np.dstack(img)
			img[img<0] = 0
			img[img>1] = 1
			if (label is not None) and adjust: self.mask = img[...,1]
			### nan to gray
			img[np.isnan(img)] = 0#.75

			#idx = (img[...,0]<0.01) & (img[...,1]<0.01) & (img[...,2]<0.01)
			#img += idx[..., np.newaxis]*0.75

			#np.save('image.npy', (img.data[::5,::5,:]*255).astype(int))


		###deal with extent
		self.header = header
		ext = [*LinearWCS(self.header,1).extent, *(LinearWCS(self.header,2).extent/(1e3 if self.lvmap else 1))]

		print(img.shape)
		if full_res:
			### export a png of full resolution
			self.exportFullRes(header, img, label, all_zeros=all_zeros)
			return
		else:
			### zero to gray
			if img.ndim==3:
				zeros = (img==0).all(axis=2)
			else:
				zeros = img==0
			img[zeros] = all_zeros


		###deal with figure
		if self.figure is None:
			self.createFigure(**figure_kws)

		###draw
		if self.separate:
			for i in range(self.parts):
				ax = self.figure[i].add_subplot()
				if img is not None:
					im = ax.imshow(img, origin='lower', extent=ext, **kws)
					self.cmap = im.cmap

				if label is not None:
					if adjust and i==0: self.adjustLabel(ax, label)
					else: self.drawLabel(ax, label)

				if cat_kws is not None:
					self.drawCat(ax, **cat_kws)

				self.configAxes(ax, i, bottom_panel=True)
				output = 'S%i_%s_%s.%s' % (self.parts, i+1, self.suffix, self.exportformat)
				self.figure[i].savefig(output)
				self.figure[i].clear()
				print('Export %s' % output)
		else:
			for i in range(self.parts):
				ax = self.figure.add_subplot(self.parts, 1, i+1)

				if img is not None:
					im = ax.imshow(img, origin='lower', extent=ext, **kws)
					self.cmap = im.cmap

				if label is not None:
					if adjust and i==0: self.adjustLabel(ax, label)
					else: self.drawLabel(ax, label)

				if cat_kws is not None:
					self.drawCat(ax, **cat_kws)

				self.configAxes(ax, i, bottom_panel=(i==self.parts-1))
			
			#plt.show()
			
			output = 'S%i_0_%s.%s' % (self.parts, self.suffix, self.exportformat)
			self.figure.savefig(output, format=self.exportformat, dpi=self.dpi, bbox_inches='tight')# tight should not be used for full resoluiotn image
			self.figure.clear()
			plt.close(self.figure)
			print('Export %s' % output)

		#self.colorbar()



if __name__ == '__main__':

	table = 'cleanedTiles/labels_noline.txt'
	cfalut = mpl.colors.ListedColormap(fits.open('cfa.fits')[0].data)


	###II
	if 1:
		rgb_kws = dict( \
			scale='linear', \
			vmin = (0, 0, -0.3), \
			#vmax = (5, 35, 120), 	#scale='sqrt' for whole
			vmax = (7, 48, 85),  #scale='linear' for W40
			label = table, \
			)

		rgb_composite_kws = dict( \
			scale='sqrt', \
			vmin = (0, 0, -0.3), \
			vmax = (10, 35, 120), \
			label = table, \
			)

		co12_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 250, \
			p = 9, \
			label = None, \
			cmap = cfalut, \
			)

		co13_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 50, \
			p = 9, \
			label = None, \
			cmap = cfalut, \
			)

		co18_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 10, \
			p = 9, \
			label = None, \
			cmap = cfalut, \
			)




		# local/*.fits are integrated over [-30,30] km/s
		# whole/*.fits are integrated over whole velocity range (cf. L-V map)
		# cleanedTiles/*.fits are integrated over vrange corresponding to glon.
		if 0:
			###dirty map
			hdu12 = fits.open('whole/tile_U_m0.fits')[0]
			hdu13 = fits.open('whole/tile_L_m0.fits')[0]
			hdu18 = fits.open('whole/tile_L2_m0.fits')[0]
			###Do this to hide bad channel contaminations, might be problematic###
			hdu18.data[hdu18.data>hdu13.data]=0

			job = ShowTiles(suffix='RGB_Uncleaned_II')
			job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, **rgb_kws)

		if 0:
			###dirty 12CO map
			hdu12 = fits.open('whole/tile_U_m0.fits')[0]
			job = ShowTiles(suffix='12CO_Uncleaned_II')
			job.draw(hdu12.data, header=hdu12.header, **co12_kws)

			hdu13 = fits.open('whole/tile_L_m0.fits')[0]
			job = ShowTiles(suffix='13CO_Uncleaned_II')
			job.draw(hdu13.data, header=hdu13.header, **co13_kws)

			hdu18 = fits.open('whole/tile_L2_m0.fits')[0]
			job = ShowTiles(suffix='C18O_Uncleaned_II')
			job.draw(hdu18.data, header=hdu18.header, **co18_kws)

		if 0:
			###clean RGB map
			hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
			hdu13 = fits.open('cleanedTiles/tile_L_m0clip3622_noexpbase0_corr.fits')[0]
			hdu18 = fits.open('cleanedTiles/tile_L2_m0clip3622_noexpbase0_corr.fits')[0]
			
			job = ShowTiles(suffix='RGB_CLeaned_II', parts=3, quality='low', axis_off=False, exportformat='pdf')
			job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, **rgb_kws)

			#job = ShowTiles(suffix='W40', parts=1, quality='high', axis_off=True, exportformat='png')
			#job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, **rgb_kws)
			
			#job = ShowTiles(suffix='SerpensSouth2', parts=1, quality='high', axis_off=True, exportformat='png')
			#job.draw(hdu18.data+hdu13.data/10, hdu13.data, hdu12.data, header=hdu12.header, **rgb_composite_kws)

			'''
			from matplotlib.colors import LinearSegmentedColormap
			cdict = {'red':[[0, 0, 0], [.3, .2, .2], [.6, .9, .9], [1, .9, .9]], 
				'green':[[0, 0, 0], [.3, .2, .2], [.6, .9, .9], [1, 0, 0]],
				'blue':[[0, 0, 0], [.3, 1, 1], [.6, .9, .9], [1, 0, 0]]}
			kws = dict( \
				scale = 'log', \
				vmin = -0.1, \
				vmax = 320, \
				p = 300, \
				label = None, \
				cmap =  LinearSegmentedColormap('myCmap', segmentdata=cdict, N=256)
				)
			job = ShowTiles(suffix='MWISP2025', parts=1, quality='high', axis_off=True, exportformat='png', lrange=(45, 100))
			job.draw(hdu12.data, header=hdu12.header, **kws)
			'''

		if 0:
			###clean CO map
			hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
			job = ShowTiles(suffix='12CO_Cleaned_II')
			job.draw(hdu12.data, header=hdu12.header, **co12_kws)
			
			hdu13 = fits.open('cleanedTiles/tile_L_m0clip3622_noexpbase0_corr.fits')[0]
			job = ShowTiles(suffix='13CO_Cleaned_II')
			job.draw(hdu13.data, header=hdu13.header, **co13_kws)

			hdu18 = fits.open('cleanedTiles/tile_L2_m0clip3622_noexpbase0_corr.fits')[0]
			job = ShowTiles(suffix='C18O_Cleaned_II')
			job.draw(hdu18.data, header=hdu18.header, **co18_kws)
			

	### DR paper
	if 1:
		rgb_composite_kws = dict( \
			scale='log', \
			vmin = (0, 0, -0.), \
			vmax = (200*0.17*0.11, 200*0.17, 200), \
			label = None, \
			p = 99,
			)
		### int clouds from Qingzeng Yan
		hdu12 = fits.open('updateClean/CO12int.fits')[0]
		hdu13 = fits.open('updateClean/CO13intFITSMay2025.fits')[0]
		hdu18 = fits.open('updateClean/C18OintFITSMay2025.fits')[0]
		'''
		### good looking from myself
		hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
		hdu13 = fits.open('cleanedTiles/tile_L_m0clip3622_noexpbase0_corr.fits')[0]
		hdu18 = fits.open('cleanedTiles/tile_L2_m0clip3622_noexpbase0_corr.fits')[0]
		'''
		#job = ShowTiles(suffix='resolution_test_noaxis', parts=3, quality='low', axis_off=False, exportformat='png')

		job = ShowTiles(parts=3, quality='low', axis_off=False, exportformat='pdf')

		rgb_composite_kws['label'] = None
		job.suffix = 'RGB_low_nolabel'
		job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, full_res=False, all_zeros=0, **rgb_composite_kws)
		job.suffix = 'RGB_fullRes_nolabel'
		job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, full_res=True, all_zeros=0.8, **rgb_composite_kws)

		rgb_composite_kws['label'] = table
		job.suffix = 'RGB_low_label'
		job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, full_res=False, all_zeros=0, **rgb_composite_kws)
		job.suffix = 'RGB_fullRes_label'
		job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, full_res=True, all_zeros=0.8, **rgb_composite_kws)


	#Outflow and Infall
	if 0:
		import pandas as pd
		# read outflow
		dfOutflow = pd.read_csv('cleanedTiles/allTableOutflow.csv')
		for i, row in dfOutflow.iterrows(): dfOutflow.at[i, 'vr'] = eval(row['vr'])

		# Read infall
		dfInfall = pd.read_csv('cleanedTiles/allTableInfall.txt', 
						 delim_whitespace=True,
						 header=None, 
						 names=['num', 'name', 'l', 'b', 'Dist', 'Vc', 'Vb', 'dV', 'T1', 'T2', 'T3', 'NH2', 'pair'],
						 dtype={0: float, 1: str})
		for col in dfInfall.columns[2:]: dfInfall[col] = dfInfall[col].astype(float)

		
		###clean CO overlay catalog
		co12_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 100, \
			p = 9, \
			label = None, \
			cmap = 'Greys', \
			cat_kws = dict(l=dfOutflow['l'], b=dfOutflow['b'], v=dfOutflow['vr'], tag=dfOutflow['shift'], color={'B':'blue', 'R':'red'}, size={'B':8,'R':6}, label={'B':'Blue-shifted', 'R':'Red-shifted'}, alpha=0.5, edgecolor='none'),
			#cat_kws = dict(l=dfInfall['l'], b=dfInfall['b'], v=dfInfall['Vc'], tag=dfInfall['pair'], color={1:'blue', 2:'red'}, size={1:15,2:15}, marker={1:'+',2:'x'}, label={1:'$^{12}$CO & $^{13}$CO', 2:'$^{13}$CO & C$^{18}$O'}, linewidth=1),
			)
		hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
		job = ShowTiles(suffix='12CO_outflow', parts=3, quality='med', exportformat='png', lvmap=False)
		job.draw(hdu12.data, header=hdu12.header, **co12_kws)

		'''
		co12_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 20, \
			p = 99, \
			label = None, \
			cmap = 'Greys', \
			#cat_kws = dict(l=dfOutflow['l'], b=dfOutflow['b'], v=dfOutflow['vr'], tag=dfOutflow['shift'], color={'B':'blue', 'R':'red'}, label={'B':'Blue-shifted', 'R':'Red-shifted'}, linewidth=4, alpha=0.5),
			cat_kws = dict(l=dfInfall['l'], b=dfInfall['b'], v=dfInfall['Vc'], tag=dfInfall['pair'], color={1:'blue', 2:'red'}, size={1:600,2:600}, marker={1:'+',2:'x'}, label={1:'$^{12}$CO & $^{13}$CO', 2:'$^{13}$CO & C$^{18}$O'}, linewidth=10),
			)
		hdu12 = fits.open('cleanedTiles/tile_U_lvclip3623_noexpbase0_corr_to13CO.fits')[0]
		job = ShowTiles(suffix='12CO_lv_infall', parts=1, quality='low', exportformat='png', lvmap=True, vrange=[-120,200])
		job.draw(hdu12.data, header=hdu12.header, figure_kws={'lvratio':15,'fontsize':50}, **co12_kws)
		'''


	#Gemini OB1
	if 0:
		import pandas as pd

		# Read the file
		li2018 = pd.read_csv('/Users/shaobo/Work/mwisp/DeepOutflow/science/lhwInLiterature/li2018.dat', 
						 delim_whitespace=True,
						 header=None, 
						 names=['name', 'shift', 'l', 'b', 'vr'],
						 usecols=[0, 1, 2, 3, 4],
						 dtype={0: str, 1: str})
		for col in li2018.columns[2:]: li2018[col] = li2018[col].astype(float)

		dfOutflow = pd.read_csv('cleanedTiles/allTableOutflow.csv')
		for i, row in dfOutflow.iterrows(): dfOutflow.at[i, 'vr'] = eval(row['vr'])

		
		mpl.rcParams['axes.linewidth'] = 0.5
		mpl.rcParams["xtick.major.size"] = 1
		mpl.rcParams["xtick.major.width"] = 0.5
		mpl.rcParams["ytick.major.size"] = 0.5
		mpl.rcParams["ytick.major.width"] = 0.5
		###clean CO overlay catalog
		co12_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 250, \
			p = 9, \
			label = None, \
			cmap = 'Greys', \
			cat_kws = dict(l=dfOutflow['l'], b=dfOutflow['b'], v=dfOutflow['vr'], tag=dfOutflow['shift'], color={'B':'blue', 'R':'red'}, size={'B':4,'R':4}, label={'B':'Blue-shifted', 'R':'Red-shifted'}, alpha=0.9, edgecolor='none', marker={'B':'+', 'R':'+'}, linewidth=.2),
			#cat_kws = dict(l=li2018['l'], b=li2018['b'], v=li2018['vr'], tag=li2018['shift'], color={'Blue':'blue', 'Red':'red'}, size={'Blue':4,'Red':4}, label={'Blue':'Blue-shifted', 'Red':'Red-shifted'}, alpha=0.5, edgecolor='none'),
			)
		hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
		job = ShowTiles(suffix='12CO_outflow_Li2018', parts=1, quality='high', exportformat='png', lvmap=False, lrange=[186.2, 195.3], brange=[-3.75,2.75])
		job.draw(hdu12.data, header=hdu12.header, figure_kws={'fontsize':6}, **co12_kws)


	#W345
	if 0:
		import pandas as pd

		# Read the file
		li2019 = pd.read_fwf('/Users/shaobo/Work/mwisp/DeepOutflow/science/lhwInLiterature/li2019.dat', 
						skiprows=36,
						widths = (3,8,7,6,6,6,6),
						names = ['num', 'l', 'b', 'Bl', 'Bu', 'Rl', 'Ru'],
						na_values = [''])
		li2019['shift'] = ''
		li2019['vr'] = 0.0
		li2019['shift'][np.isfinite(li2019['Bl']) & np.isnan(li2019['Rl'])] = 'Blue'
		li2019['shift'][np.isnan(li2019['Bl']) & np.isfinite(li2019['Rl'])] = 'Red'
		li2019['shift'][np.isfinite(li2019['Bl']) & np.isfinite(li2019['Rl'])] = 'Purple'

		#dfOutflow = pd.read_csv('cleanedTiles/allTableOutflow_alsoBad.csv')
		#for i, row in dfOutflow.iterrows(): dfOutflow.at[i, 'vr'] = eval(row['vr'])

		
		mpl.rcParams['axes.linewidth'] = 0.5
		mpl.rcParams["xtick.major.size"] = 1
		mpl.rcParams["xtick.major.width"] = 0.5
		mpl.rcParams["ytick.major.size"] = 0.5
		mpl.rcParams["ytick.major.width"] = 0.5
		###clean CO overlay catalog
		co12_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 150, \
			p = 9, \
			label = None, \
			cmap = 'Greys', \
			#cat_kws = dict(l=dfOutflow['l'], b=dfOutflow['b'], v=dfOutflow['vr'], tag=dfOutflow['shift'], color={'B':'blue', 'R':'red'}, size={'B':4,'R':4}, label={'B':'Blue-shifted', 'R':'Red-shifted'}, alpha=0.5, edgecolor='none'),
			cat_kws = dict(l=li2019['l'], b=li2019['b'], v=li2019['vr'], tag=li2019['shift'], color={'Blue':'blue', 'Red':'red', 'Purple':'purple'}, size={'Blue':4,'Red':4, 'Purple':4}, label={'Blue':'Blue-shifted', 'Red':'Red-shifted', 'Purple':'Bipolar'}, alpha=0.5, edgecolor='none'),
			)
		hdu12 = fits.open('cleanedTiles/tile_U_m0clip3623_noexpbase0_corr.fits')[0]
		job = ShowTiles(suffix='12CO_outflow_Li2019', parts=1, quality='high', exportformat='png', lvmap=False, lrange=[129.75, 140.25], brange=[-5.25,5.25])
		job.draw(hdu12.data, header=hdu12.header, figure_kws={'fontsize':6}, **co12_kws)


	###LV
	if 1:
		rgb_lv_kws = dict( \
			scale = 'sqrt', \
			vmin = (0, 0, 0), \
			vmax = (0.4, 2.8, 9.6), \
			)

		co12_lv_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 20, \
			p = 99, \
			cmap = cfalut, \
			)
		#	label = table, \

		co13_lv_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 5, \
			p = 99, \
			cmap = cfalut, \
			)

		co18_lv_kws = dict( \
			scale = 'log', \
			vmin = 0, \
			vmax = 1, \
			p = 99, \
			cmap = cfalut, \
			)

		if 0:
			###dirty map resample
			#Resample velocity axis in LVMAP of 12CO/C18O to align with 13CO
			f12 = 'lvmapgoodlooking/tile_U_lvmap.fits'
			f13 = 'lvmapgoodlooking/tile_L_lvmap.fits'
			f18 = 'lvmapgoodlooking/tile_L2_lvmap.fits'
			resample_lvmap(f12, f13, 'lvmapgoodlooking/tile_U_lvmap_to13CO.fits')
			resample_lvmap(f18, f13, 'lvmapgoodlooking/tile_L2_lvmap_to13CO.fits')

		if 0:
			###dirty RGB lv map
			hdu12 = fits.open('lvmapgoodlooking/tile_U_lvmap_to13CO.fits')[0]
			hdu13 = fits.open('lvmapgoodlooking/tile_L_lvmap.fits')[0]
			hdu18 = fits.open('lvmapgoodlooking/tile_L2_lvmap_to13CO.fits')[0]

			job = ShowTiles(parts=1, lvmap=True, suffix='RGB_Uncleaned_LV')
			job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, **rgb_lv_kws)

		if 0:
			###dirty 12CO lv map
			hdu12 = fits.open('lvmapgoodlooking/tile_U_lvmap.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='12CO_Uncleaned_LV')
			job.draw(hdu12.data, header=hdu12.header, **co12_lv_kws)

			hdu13 = fits.open('lvmapgoodlooking/tile_L_lvmap.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='13CO_Uncleaned_LV')
			job.draw(hdu13.data, header=hdu13.header, **co13_lv_kws)
			
			hdu18 = fits.open('lvmapgoodlooking/tile_L2_lvmap.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='C18O_Uncleaned_LV')
			job.draw(hdu18.data, header=hdu18.header, **co18_lv_kws)

		if 0:
			###clean lv map resample
			#Resample velocity axis in LVMAP of 12CO/C18O to align with 13CO
			f12 = 'cleanedTiles/tile_U_lvclip3623_noexpbase0_corr.fits'
			f13 = 'cleanedTiles/tile_L_lvclip3622_noexpbase0_corr.fits'
			f18 = 'cleanedTiles/tile_L2_lvclip3622_noexpbase0_corr.fits'
			resample_lvmap(f12, f13, 'cleanedTiles/tile_U_lvclip3623_noexpbase0_corr_to13CO.fits')
			resample_lvmap(f18, f13, 'cleanedTiles/tile_L2_lvclip3622_noexpbase0_corr_to13CO.fits')

		if 0:
			###clean RGB lv map
			hdu12 = fits.open('cleanedTiles/tile_U_lvclip3623_noexpbase0_corr_to13CO.fits')[0]
			hdu13 = fits.open('cleanedTiles/tile_L_lvclip3622_noexpbase0_corr.fits')[0]
			hdu18 = fits.open('cleanedTiles/tile_L2_lvclip3622_noexpbase0_corr_to13CO.fits')[0]

			job = ShowTiles(parts=1, lvmap=True, suffix='RGB_Cleaned_LV')
			job.draw(hdu18.data, hdu13.data, hdu12.data, header=hdu12.header, **rgb_lv_kws)

		if 0:
			###clean CO lv map
			hdu12 = fits.open('cleanedTiles/tile_U_lvclip3623_noexpbase0_corr.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='12CO_Cleaned_LV')
			job.draw(hdu12.data, header=hdu12.header, **co12_lv_kws)
			
			hdu13 = fits.open('cleanedTiles/tile_L_lvclip3622_noexpbase0_corr.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='13CO_Cleaned_LV')
			job.draw(hdu13.data, header=hdu13.header, **co13_lv_kws)
			
			hdu18 = fits.open('cleanedTiles/tile_L2_lvclip3622_noexpbase0_corr.fits')[0]
			job = ShowTiles(parts=1, lvmap=True, suffix='C18O_Cleaned_LV')
			job.draw(hdu18.data, header=hdu18.header, **co18_lv_kws)
			


	###rms
	if 1:
		rms12_kws = dict( \
			scale = 'lin', \
			vmin=0.3, \
			vmax=0.7, \
			label=None, \
			cmap='rainbow', \
			)
		rms13_kws = dict( \
			scale = 'lin', \
			vmin=0.15, \
			vmax=0.375, \
			label=None, \
			cmap='rainbow', \
			)
		rms18_kws = dict( \
			scale = 'lin', \
			vmin=0.15, \
			vmax=0.375, \
			label=None, \
			cmap='rainbow', \
			)

		if 0:
			###dirty RMS
			hdu12 = fits.open('whole/tile_U_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='12CO_Uncleaned_rms')
			job.draw(hdu12.data, header=hdu12.header, **rms12_kws)

			hdu13 = fits.open('whole/tile_L_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='13CO_Uncleaned_rms')
			job.draw(hdu13.data, header=hdu13.header, **rms13_kws)

			hdu18 = fits.open('whole/tile_L2_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='C18O_Uncleaned_rms')
			job.draw(hdu18.data, header=hdu18.header, **rms18_kws)

		if 0:
			###clean RMS
			hdu12 = fits.open('cleanedTiles/tile_U_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='12CO_Cleaned_rms')
			job.draw(hdu12.data, header=hdu12.header, **rms12_kws)

			hdu13 = fits.open('cleanedTiles/tile_L_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='13CO_Cleaned_rms')
			job.draw(hdu13.data, header=hdu13.header, **rms13_kws)

			#hdu18 = fits.open('cleanedTiles/tile_L2_rms.fits')[0]
			hdu18 = fits.open('cleanedTiles/tile_L2_rms.fits')[0]
			job = ShowTiles(rmsmap=True, suffix='C18O_Cleaned_rms')
			job.draw(hdu18.data, header=hdu18.header, **rms18_kws)




