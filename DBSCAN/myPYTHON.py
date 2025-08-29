
# this File contains my own defined functions, which is very customed

# put this as the only version
from __future__ import print_function

import gc
import sys

import radio_beam
from scipy.odr import *
import numpy as np
from astropy.table import Table, Column,vstack

from astropy.io import fits

from matplotlib import pyplot as plt

from astropy.wcs import WCS
import os
import math
import os.path
# import pywcsgrid2
from matplotlib import rc

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from numpy import mean, sqrt, square

import matplotlib as mpl

from subprocess import call

from astropy import units as u
from spectral_cube import SpectralCube
from astropy.convolution import Gaussian1DKernel

from decimal import Decimal,ROUND_HALF_UP
#from pvextractor import extract_pv_slice
#from pvextractor import  Path as ppvPath

from pathlib import Path
paperPath = "/Users/qzyan/Desktop/UrsaMajorApJS/"


"""
Version 3.0

Adding a function to select complete clouds from cat log 

"""




def model(p, x):
	a, b = p
	return a + b * x


def fitLinearODR(x, y, x_err, y_err):
	# Create a model for fitting.

	quad_model = Model(model)

	# Create a RealData object using our initiated data from above.
	data = RealData(x, y, sx=x_err, sy=y_err)

	# Set up ODR with the model and data.
	odr = ODR(data, quad_model, beta0=[1, 1])

	# Run the regression.
	out = odr.run()
	# out.pprint() b

	return out




class myFITS:
	"This is fits is my own class of fits"
	fitsDATA = None
	fitsHeader = None
	fitsName = None
	fitsPath = None
	thisFITS = None
	rmsCO12 = 0.5
	R0 = 8.178  # 34 the background would be changed #  2019A&A...625L..10G

	dataPath="./data/"
	figurePath="./figures/"

	#### some frequently used definitions==================
	IDcol = "_idx" ### used to much
	col12COID="12COID"
	#### =================================
	unit_flux=r"K km s$^{-1}$ arcmin$^{-2}$"
	unit_Vlsr=r"$\mathit{V}_{\rm LSR}$ (km s$^{-1}$)"
	unit_Tmb=r"$\mathit{T}_{\rm mb}$ (K)"
	def __init__(self, FITSname=None):
		if FITSname:  # with fits input
			print("An object processing {} created!".format(FITSname))
			self.fitsDATA, self.fitsHeader = self.readFITS(FITSname)
			self.fitsPath, self.fitsName = os.path.split(FITSname)
			if self.fitsPath == "":
				self.fitsPath = "./"

			self.thisFITS = FITSname


		else:
			pass

	# print "No input FITS is provided for myFITS"
	# If there is a paramters input, automatically read this FITS
	# and some paramters will be automatically figured out

	def checkDataFigurePath(self):
		self.checkPath(self.dataPath)
		self.checkPath(self.figurePath)

	@staticmethod
	def lineFit(X1, Y1, errorX, errorY):
		"""
		Fitting a line
		"""

		out1 = fitLinearODR(X1, Y1, errorX, errorY)

		intercept, slope = out1.beta  # fitobj.params

		interceptError, std_err = out1.sd_beta

		return [slope, intercept, std_err, interceptError]

	@staticmethod
	def getDeltaV(header):

		return abs(header["CDELT3"]/1000.)

	@staticmethod
	def getVoxValue(data, dataWCS, LBV):
		"""
		This function is used to get a voxel value from a data cube

		v, km/s
		"""
		l, b, v = LBV

		z, y, x = data.shape

		try:
			indexX, indexY, indexZ = dataWCS.wcs_world2pix(l, b, v * 1000, 0)
		except:
			indexX, indexY, indexZ, a = dataWCS.wcs_world2pix(l, b, v * 1000, 0, 0)

		indexX = int(round(indexX))
		indexY = int(round(indexY))
		indexZ = int(round(indexZ))

		# print indexX,indexY,indexZ

		# print x,y,z
		# print indexX,indexY,indexZ

		if indexX >= x or indexY >= y or indexZ >= z:
			return np.NaN

		return data[indexZ, indexY, indexX]

	@staticmethod
	def weighted_avg_and_std(values, weights):
		"""
		Return the weighted average and standard deviation.

		values, weights -- Numpy ndarrays with the same shape.
		"""

		weights=weights/np.sum(weights)
		average = np.average(values, weights=weights)
		# Fast and numerically precise:
		variance = np.average((values - average) ** 2, weights=weights)

		# if variance<0:
		# print weights

		return (average, math.sqrt(variance))


	@staticmethod
	def showImg(data2D):
		"""

		Returns
		-------

		"""
		plt.imshow(data2D,origin="lower")
		plt.show()
		plt.close()
	@staticmethod
	def sumAroundFITS2D(data2D):
		"""

		Returns
		-------

		"""


		extendMask = np.pad(data2D, ((1, 1), (1, 1)), "constant", constant_values=0)

		sum = extendMask[1:-1, 1:-1] + extendMask[0:-2, 1:-1] + extendMask[2:, 1:-1]
		sum = sum + extendMask[1:-1, 0:-2] + extendMask[1:-1, 2:]

		return  sum


	@staticmethod
	def getHistCenter(histEdge ):
		"""

		"""

		return (histEdge[0:-1]+histEdge[1:])/2


	@staticmethod
	def getEdgeMask2D(dataCube):
		"""

		Returns
		-------

		"""

		# remove all internal regions
		dataMask = dataCube > 0
		dataMask = dataMask * 1

		extendMask = np.pad(dataMask, ((1, 1), (1, 1)), "constant", constant_values=0)

		sum = extendMask[1:-1, 1:-1] + extendMask[0:-2, 1:-1] + extendMask[2:, 1:-1]
		sum = sum + extendMask[1:-1, 0:-2] + extendMask[1:-1, 2:]

		edgeMask = np.logical_and(dataMask == 1, sum <  5)

		return edgeMask * 1

	@staticmethod
	def getCompleteMC(MCcat,completeCol="complete"):
		"""
		#remove incomplete clouds in LB
		"""


		newCat = MCcat[MCcat[completeCol]>0.5]

		return newCat


	def convertMask( self,maskFITS ,targetFITS, keepMaskID=False,outFITS=None):
		"""

		#### convert CO12 mask to CO12 mask

		"""
		dataMask,headMask= self.readFITS(maskFITS )
		targetData,targetHead= self.readFITS( targetFITS )


		indexOfCO12pix = np.where(dataMask > 0)  ####

		wcsCO12 = WCS(headMask)
		wcsCO13 = WCS(targetHead)

		iZ, iY, iX = indexOfCO12pix

		l12, b12, v12 = wcsCO12.wcs_pix2world(iX, iY, iZ, 0)
		iX13, iY13, iZ13 = wcsCO13.wcs_world2pix(l12, b12, v12, 0)
		iX13 = np.int64(np.rint(iX13))
		iY13 = np.int64(np.rint(iY13))
		iZ13 = np.int64(np.rint(iZ13))
		dataCO13Mask = np.zeros_like(targetData)
		if keepMaskID:
			dataCO13Mask[(iZ13, iY13, iX13)] = dataMask[indexOfCO12pix]
		else:

			dataCO13Mask[(iZ13, iY13, iX13)] = 1


		if outFITS is not None:
			fits.writeto(outFITS,dataCO13Mask,header=targetHead,overwrite= True )

		return dataCO13Mask

	def assignCloudIDto13CO(self,CO13ClumpTableFile,cloudRegionMaskAtCO13,saveTag="" ):

		CO13CloudMask =cloudRegionMaskAtCO13 #"/perseus/qzyanDisk/Process2025/CO13/DBSCANresults/cloudRegionAtCO13_withCO12ID.fits"
		CO13ClumpTableFile =CO13ClumpTableFile #"/perseus/qzyanDisk/Process2025/CO13/DBSCANresults/catCO13_Nov_L_AGC_edgedbscanS2P4Con1catCO13May2025_Clean.fit"

		CO13ClumpTB = Table.read(CO13ClumpTableFile)

		dataCO13Mask, headCO13Mask = self.readFITS(CO13CloudMask)
		catCO13WithCO12ID = CO13ClumpTB.copy()
		catCO13WithCO12ID[self.col12COID] = np.zeros_like(catCO13WithCO12ID["_idx"])

		for eachClump in catCO13WithCO12ID:
			clumpL = eachClump["peakL"]
			clumpB = eachClump["peakB"]
			clumpV = eachClump["peakV"]

			cloudIndex = dataCO13Mask[
				clumpV, clumpB, clumpL]  # getVoxValue(dataCO12Mask,wcsCO12,[clumpL,clumpB,clumpV])

			eachClump[self.col12COID] = cloudIndex

		saveCat=os.path.basename(CO13ClumpTableFile)
		saveCat = os.path.splitext(saveCat)[0]
		saveCat = saveCat+"_WithCO12ID_{}.fit".format( saveTag )
		catCO13WithCO12ID.write( saveCat, overwrite=True)

	@staticmethod
	def convertCO12MaskTOCO13Mask( CO12Mask, headCO12,dataCO13,headCO13,keepCO12ID=False):
		"""

        #### convert CO12 mask to CO12 mask

        """



		indexOfCO12pix = np.where(CO12Mask > 0)  ####

		wcsCO12 = WCS(headCO12)
		wcsCO13 = WCS(headCO13)

		iZ, iY, iX = indexOfCO12pix

		l12, b12, v12 = wcsCO12.wcs_pix2world(iX, iY, iZ, 0)
		iX13, iY13, iZ13 = wcsCO13.wcs_world2pix(l12, b12, v12, 0)
		iX13 = np.int64(np.rint(iX13))
		iY13 = np.int64(np.rint(iY13))
		iZ13 = np.int64(np.rint(iZ13))
		dataCO13Mask = np.zeros_like(dataCO13)
		if keepCO12ID:
			dataCO13Mask[(iZ13, iY13, iX13)] = CO12Mask[indexOfCO12pix]
		else:

			dataCO13Mask[(iZ13, iY13, iX13)] = 1

		return dataCO13Mask

	@staticmethod
	def getPixValue(data, head, LB):
		"""
		This function is used to get a voxel value from a data cube

		v, km/s
		"""
		l, b = LB

		y, x = data.shape
		dataWCS = WCS(head)

		try:
			indexX, indexY = dataWCS.wcs_world2pix(l, b, 0)
		except:
			indexX, indexY, indexZ = dataWCS.wcs_world2pix(l, b, 0, 0)

		indexX = int(round(indexX))
		indexY = int(round(indexY))

		if indexX >= x or indexY >= y:
			return np.NaN

		return data[indexY, indexX]

	@staticmethod
	def downTo2D(fitsFile, outPUT=None, overwrite=True):
		"""
		some files creaseted by Miriad is 3D, and there is only one axis in the third

		this functino is used to transformd 3D to 2d
		"""

		if str.strip(fitsFile) == '':
			print("Empty file, quitting....")
			return

		if outPUT is None:
			writeName = "temp2DFile.fits"  # +fitsFile

		else:
			writeName = outPUT

		fileExist = os.path.isfile(writeName)

		if overwrite and fileExist:
			os.remove(writeName)

		if not overwrite and fileExist:
			print("File exists, quitting...")
			return

		# read file
		hdu = fits.open(fitsFile)[0]

		head = hdu.header
		# wmap=WCS(hdu.header)
		data = hdu.data

		data = data[0]

		del head["CRPIX3"]
		del head["CDELT3"]
		del head["CRVAL3"]
		del head["CTYPE3"]

		# writefits

		fits.writeto(writeName, data, header=head)

		return fits.open(writeName)[0]




	@staticmethod
	def roundToInt(someArray):

		"""
		round to nearest integer
		"""

		convertInt = np.rint(someArray)
		convertInt = list(map(int, convertInt))

		return convertInt
	@staticmethod
	def getLB(fitsName, prefix, lineCode):
		"""

        """

		baseName = os.path.basename(fitsName)

		if prefix=="":
			lbstr =  baseName
		else:
			lbstr = baseName.split(prefix)[-1]

		lbstr = lbstr.split(lineCode)[0]

		if "+" in lbstr:
			lStr, bStr = lbstr.split("+")
			l = np.float64(lStr) / 10.
			b = np.float64(bStr) / 10.
		else:
			lStr, bStr = lbstr.split("-")

			l = np.float64(lStr) / 10.
			b = -np.float64(bStr) / 10.

		return l, b

	@staticmethod
	def getFITSName(l, b, prefix="NC", lineCode="L"):
		"""

        :param l:
        :param b:
        :return:
        """
		#### merge a cente LB

		lGrids = np.arange(10, 230, 0.5)
		bGrids = np.arange(-5, 5.5, 0.5)

		# pick L

		subL = np.abs(lGrids - l)
		l = lGrids[subL.argmin()]
		subb = np.abs(bGrids - b)
		b = bGrids[subb.argmin()]

		l = l * 10
		b = b * 10
		strL = str(int(l))
		strB = str(int(abs(b)))
		strL = strL.zfill(4)
		strB = strB.zfill(3)

		if b < 0:

			fitsName = "{}{}-{}{}.fits".format(prefix, strL, strB, lineCode)
			fitsNameRms = "{}{}-{}{}_rms.fits".format(prefix, strL, strB, lineCode)

		else:
			fitsName = "{}{}+{}{}.fits".format(prefix, strL, strB, lineCode)
			fitsNameRms = "{}{}+{}{}_rms.fits".format(prefix, strL, strB, lineCode)

		return fitsName, fitsNameRms
	@staticmethod
	def newRound( a):
		if type(a)==list or type(a) == np.ndarray:

			return map(int,np.rint(a) )

		return int(np.rint(a) )

	@staticmethod
	def round_int( a):

		"""
        only for this purpuse
        :param a:
        :return:
        """

		a = float(a)
		middleA = Decimal(a).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
		return int(middleA)

	@staticmethod
	def getLBVbyIndex(wcs, XYZ):
		X, Y, Z = XYZ
		X = int(round(X));
		Y = int(round(Y));
		Z = int(round(Z));

		l, b, v = wcs.all_pix2world(X, Y, Z, 0)[0:3]

		return [l, b, v]

	def getVoxValueByIndex(self, data, XYZ):
		"""
		This function is used to get a voxel value from a data cube

		v, km/s

		"""
		X, Y, Z = XYZ
		X = int(round(X))
		Y = int(round(Y))
		Z = int(round(Z))

		x, y, z = data.shape

		# print indexX,indexY,indexZ

		# print x,y,z
		# print indexX,indexY,indexZ

		if X >= z or Y >= y or Z >= z or X < 0 or Y < 0 or Z < 0:
			return np.NaN

		return data[Z][Y][X]


	def smoothVelAxisNoSampling(self,fitsName, smFactor, outPutName=None, savePath=None, onlyName=False   ):
		"""
		Resample a data cube along the velocity
		only used for MWISP

		:param dadta:
		:param dataHeader:
		:param targetVelResolution: in unites of km/s
		:param saveName:
		:return:
		"""

		print("Starting to regrid the velocity axis, this may take a little while...")
		if outPutName is None:
			saveTag = "velSM_{:.0f}".format(smFactor)
			saveName = self.addSuffix(fitsName, saveTag)
			outPutName = os.path.split(saveName)[1]

		if savePath is not None:
			outPutName = os.path.join(savePath, outPutName)

		if onlyName:
			return outPutName

		if smFactor == 1:
			self.converto32bit(fitsName, outPutName)  ####  there is no nned to process
			return outPutName
		if smFactor < 1:
			print("Errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrooooooooooooooor! The smooth factor need to be at least 1")
			sys.exit()


		#####################################
		cube = SpectralCube.read(fitsName)

		fwhm_factor = np.sqrt(8 * np.log(2))

		# get current reslution
		velPix = cube.header["CDELT3"] / 1000.  # in km/s
		current_resolution = velPix * u.km / u.s


		target_resolution = smFactor *  current_resolution

		pixel_scale = velPix * u.km / u.s
		gaussian_width = ((target_resolution ** 2 - current_resolution ** 2) ** 0.5 /  pixel_scale / fwhm_factor)

		kernel = Gaussian1DKernel(gaussian_width.value)
		new_cube = cube.spectral_smooth(kernel)
		new_cube.write(outPutName, overwrite=True, format="fits")

		self.converto32bit(outPutName, outPutName) #### to save space, it is needed?


		return outPutName
		########
		newDV = targetVelResolution  # km/s

		vAxis = cube.spectral_axis.value / 1000.  # in

		minV = np.min(vAxis)
		# minV=int(minV)-1

		maxV = np.max(vAxis)
		# maxV=int(maxV)+1

		#
		if maxV < 0:
			new_axis = np.arange(0, abs(minV), newDV)
			new_axis = - new_axis
			new_axis = new_axis[::-1]

		elif minV > 0:
			new_axis = np.arange(0, maxV, newDV)

		else:

			new_axisPart1 = np.arange(0, maxV, newDV)
			new_axisPart2 = np.arange(newDV, abs(minV), newDV)
			new_axisPart2 = -new_axisPart2
			new_axisPart2 = new_axisPart2[::-1]

			new_axis = np.concatenate([new_axisPart2, new_axisPart1])

		new_axis = new_axis * 1000 * u.m / u.s
		new_axis = new_axis[new_axis >= 1000 * np.min(vAxis) * u.m / u.s]
		new_axis = new_axis[new_axis <= 1000 * np.max(vAxis) * u.m / u.s]

		if mimicFITS is not None:
			cubeMimic = SpectralCube.read(mimicFITS)
			new_axis = cubeMimic.spectral_axis

		interp_Cube = new_cube.spectral_interpolate(new_axis, suppress_smooth_warning=True)

		interp_Cube.write(saveName, overwrite=True, format="fits")

		print("Smooting the velocity axis done!")


	@staticmethod
	def smoothVelAxis(fitsName, targetVelResolution, saveName, mimicFITS=None):
		"""
		Resample a data cube along the velocity
		only used for MWISP

		:param dadta:
		:param dataHeader:
		:param targetVelResolution: in unites of km/s
		:param saveName:
		:return:
		"""

		print("Starting to regrid the velocity axis, this may take a little while...")

		cube = SpectralCube.read(fitsName)

		fwhm_factor = np.sqrt(8 * np.log(2))

		# get current reslution
		velPix = cube.header["CDELT3"] / 1000.  # in km/s
		current_resolution = velPix * u.km / u.s

		if mimicFITS is not None:
			cubeMimic = SpectralCube.read(mimicFITS)
			targetVelResolution = cubeMimic.header["CDELT3"] / 1000. + 1

		target_resolution = targetVelResolution * u.km / u.s
		pixel_scale = velPix * u.km / u.s
		gaussian_width = ((target_resolution ** 2 - current_resolution ** 2) ** 0.5 /
						  pixel_scale / fwhm_factor)

		kernel = Gaussian1DKernel(gaussian_width.value)
		new_cube = cube.spectral_smooth(kernel)

		newDV = targetVelResolution  # km/s

		vAxis = cube.spectral_axis.value / 1000.  # in

		minV = np.min(vAxis)
		# minV=int(minV)-1

		maxV = np.max(vAxis)
		# maxV=int(maxV)+1

		#
		if maxV < 0:
			new_axis = np.arange(0, abs(minV), newDV)
			new_axis = - new_axis
			new_axis = new_axis[::-1]

		elif minV > 0:
			new_axis = np.arange(0, maxV, newDV)

		else:

			new_axisPart1 = np.arange(0, maxV, newDV)
			new_axisPart2 = np.arange(newDV, abs(minV), newDV)
			new_axisPart2 = -new_axisPart2
			new_axisPart2 = new_axisPart2[::-1]

			new_axis = np.concatenate([new_axisPart2, new_axisPart1])

		new_axis = new_axis * 1000 * u.m / u.s
		new_axis = new_axis[new_axis >= 1000 * np.min(vAxis) * u.m / u.s]
		new_axis = new_axis[new_axis <= 1000 * np.max(vAxis) * u.m / u.s]

		if mimicFITS is not None:
			cubeMimic = SpectralCube.read(mimicFITS)
			new_axis = cubeMimic.spectral_axis

		interp_Cube = new_cube.spectral_interpolate(new_axis, suppress_smooth_warning=True)

		interp_Cube.write(saveName, overwrite=True, format="fits")

		print("Smooting the velocity axis done!")




	def smoothSpaceFITS(self, fitsName, rawBeam, factor, outPutName=None,savePath=None,onlyName=False ):  # arcmin

		"""
		fitsName, rawBeam, the
		:param fitsName:
		:param rawBeam:
		:param factor:
		:param outPutName:
		:return:
		"""
		
		if outPutName is None:
			saveTag =  "beamSM_{:.0f}".format(factor)
			saveName = self.addSuffix(fitsName,saveTag)
			outPutName= os.path.split(saveName)[1]
		if savePath is not None:
			outPutName=os.path.join(savePath,outPutName)
		
		if onlyName:
			return outPutName

		if factor==1:
			self.converto32bit(fitsName, outPutName)  ####  there is no nned to process
			return outPutName
		if factor<1:

			print("Errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrooooooooooooooor! The smooth factor need to be at least 1")
			sys.exit()
		#### use spectral cube

		rawBeamSizeCO12 =  rawBeam
		# mwispBEAM =  49./60.

		rawCOFITS =  fitsName

		# smooth the data

		processCube = SpectralCube.read(rawCOFITS)
		processCube.allow_huge_operations = True

		RawBeam = radio_beam.Beam(major=rawBeamSizeCO12 * u.arcmin, minor=rawBeamSizeCO12 * u.arcmin, pa=0 * u.deg)
		processCube.beam = RawBeam
		#########################################################
		print("Smoothing beam  by factor ", factor)
		targetBeam = factor * rawBeamSizeCO12

		beamTarget = radio_beam.Beam(major=targetBeam * u.arcmin, minor=targetBeam * u.arcmin, pa=0 * u.deg)

		new_cube = processCube.convolve_to(beamTarget)

		#saveName = self.getMWISPbeamSMFileName(factors)


		new_cube.write(outPutName, overwrite=True)

		self.converto32bit(outPutName, outPutName) #### to save space, it is needed?

		return outPutName
		# dataNew=COdata.copy()

		# the pixel resolution

		# resolCO=0.125 #deg each pixe

		# convert RawBeam in arcmin to degree, calculate the size with pixels

		rawBeamDegree = rawBeam / 60.
		resultBeamDegree = resultBeam / 60.

		resolX = abs(dataHeader["CDELT1"])  # degree
		resolY = abs(dataHeader["CDELT2"])  # degree

		rawPixForX = rawBeamDegree / resolX / np.cos(np.radians(38))  #
		# dut to the shrink of galactic longitude, the actually beam is a little bit larger
		# use a 38 degree, what does thsi mean?
		#
		rawPixForY = rawBeamDegree / resolY

		resultPixForX = resultBeamDegree / resolX / np.cos(np.radians(38))  #
		resultPixForY = resultBeamDegree / resolY

		gaussianBeamX = (resultPixForX ** 2 - rawPixForX ** 2) ** 0.5
		gaussianBeamY = (resultPixForY ** 2 - rawPixForY ** 2) ** 0.5

		# the gaussianBeam is FWHM, the gaussian kernel should use sgima
		sigmaX = gaussianBeamX / 2.355
		sigmaY = gaussianBeamY / 2.355

		# creat an eliptical Gauss

		# print rawPixForX,rawPixForY
		# print resultPixForX,resultPixForY
		# print sigmaX,sigmaY

		g2 = Gaussian2D(10, x_stddev=sigmaX, y_stddev=sigmaY)
		smoothKernel = Model2DKernel(g2, x_size=17, y_size=17)

		smoothKernel.normalize()

		# print smoothKernel.array

		# smooth the data using this kernel

		if len(data.shape) == 3:

			z, y, x = data.shape

			for i in range(z):
				data[i] = convolve(data[i], smoothKernel)

		if len(data.shape) == 2:
			data = convolve(data, smoothKernel)
		# written the fits

		# CO12[0].header["CTYPE3"]="VELO-LSR"
		# CO12.writeto("CO12SM20.fits")
		# outPutName
		# hdu = pyfits.PrimaryHDU(data)
		os.remove(outPutName)
		fits.writeto(outPutName, data, dataHeader)

	def getEmptyTB(self, tb):
		"""

		:param tb:
		:return:
		"""
		newTB = tb.copy()
		newTB.add_row()
		newTB = newTB[-1:]
		newTB.remove_row(0)

		return newTB

	def getAverageSpec(self, fitsFile, path="./", cores=None):
		"""
		This function is dedicated to get the average spectrum for the CO lines

		#Basicly, we only care about the spectra at the peak posion of the cores

		#The index of peak position given by Duchamp is from 0.

		"""
		# read the file
		cores = self.getCoreTable()

		COdata, COheader = self.readFITS(path + fitsFile)

		# print len(cores)
		avgSpec = 0

		for eachCore in cores:
			# l,b= eachCore["GLON_deg"],eachCore["GLAT_deg"]
			# spectrum,vs =self.getSpectraByLB( COdata,COheader,l,b)

			X, Y = eachCore["X_peak"], eachCore["Y_peak"]
			spectrum, vs = self.getSpectraByIndex(COdata, COheader, int(X), int(Y))

			avgSpec = avgSpec + spectrum
		# print l,b,spectrum[0]
		avgSpec = avgSpec / 1. / len(cores)

		if 0:
			l, b = cores[0]["GLON_deg"], cores[0]["GLAT_deg"]

			avgSpec, vs = self.getSpectraByLB(COdata, COheader, l, b)

		if 0:
			fig, ax = plt.subplots()
			ax.plot(vs, avgSpec)
			plt.show()

		return avgSpec, vs

	def box(self, centerL, centerB, lSize, bSize, dummy=0):
		"""
		return lRange and B Range
		"""

		lSize = lSize / 3600.
		bSize = bSize / 3600.

		return [centerL - lSize / 2., centerL + lSize / 2.], [centerB - bSize / 2., centerB + bSize / 2.]

	def getAverageSpecByLBrange(self, fitsFile, lRange, bRange):

		"""
		calculate the average fitsFITS in within the lRange and bRnage
		"""
		# COdata,COheader=self.readFITS( fitsFile)

		cropedFITS = "croped.fits"  # temperature

		self.cropFITS(fitsFile, outFITS=cropedFITS, Vrange=None, Lrange=lRange, Brange=bRange, overWrite=True)




		cropData, cropHeader = self.readFITS(cropedFITS)

		# average the spectral

		averageSpectral = np.average(cropData, axis=(1, 2))

		spectrum, vs = self.getSpectraByIndex(cropData, cropHeader, 0, 0)

		return averageSpectral, vs

	def mytrim(self, d, vmin, vmax):
		dd = (d + vmin) / (vmin + vmax)
		return np.clip(dd, 0, 1)

	@staticmethod
	def convert4DTo3D(header):

		header = header.copy()
		print("Reducing to 3D....")

		header["NAXIS"] = 3

		try:
			del header["NAXIS4"]

			del header["CRPIX4"]
			del header["CDELT4"]
			del header["CRVAL4"]
			del header["CTYPE4"]
			del header["CROTA4"]

		except:
			pass

		return header



	def intFITSall(self,FITSFile):
		"""

		:param FITSFile:
		:return:
		"""

		cubeData, cubeHead = myFITS.readFITS(FITSFile)
		dv = cubeHead["CDELT3"] / 1000.  # to km/s
		return np.sum(cubeData,axis= 0 )*dv

	def intFITS(self, FITSFile, vRange=None, overwrite=True, saveName=None,savePath=None,returnIntRange=False):
		"""
		# simple integration
		:param self:
		:param FITSFile:
		:param vRange:
		:param overwrite:
		:param saveName:
		:return:
		"""

		if vRange is not None:
			minV = min(vRange)
			maxV = max(vRange)

			fitsBaseName = os.path.basename(FITSFile)

			vRange = [minV, maxV]

		if saveName is None:
			saveName = "int_{:.2f}_{:.2f}".format(minV, maxV) + fitsBaseName

		if savePath is not None:

			saveName=os.path.join(savePath,saveName)


		cubeData, cubeHead = myFITS.readFITS(FITSFile)
		wcs = WCS(cubeHead,naxis=3)
		dv = abs(cubeHead["CDELT3"] )/ 1000.  # to km/s

		Nz, Ny, Nx = cubeData.shape

		l0, b0,v0 = wcs.wcs_pix2world(0, 0, 0, 0)

		if vRange is not None:
			a, a, indexV0 = wcs.wcs_world2pix(l0, b0, vRange[0] * 1000, 0)

			a, a, indexV1 = wcs.wcs_world2pix(l0, b0, vRange[1] * 1000, 0)

			indexV0, indexV1 = list(map(np.round, [indexV0, indexV1]))
			indexV0, indexV1 = list(map(int, [indexV0, indexV1]))

			startV = max([0, indexV0])
			endV = min([indexV1, Nz - 1])

		else:

			startV = 0
			endV = Nz

		####

		#### calculate the real integral velcoity

		a, a, velV0 = wcs.wcs_pix2world(0, 0 ,  startV , 0)
		a, a, velV1 = wcs.wcs_pix2world(0, 0 ,  endV , 0)
		actualRange= [velV0/1000., velV1/1000. ]
		print("Actually integration range ({:.2f}, {:.2f}) km/s (interval: {:.2f} )".format(velV0/1000., velV1/1000. , (velV1-velV0)/1000.)  )

		startV0=min([startV,endV])
		endV0 =max([startV,endV])

		sumData = cubeData[startV0:endV0 + 1]

		sum2D = np.nansum(sumData, axis=0, dtype=float)
		sum2D = sum2D * dv

		if overwrite:

			fits.writeto(saveName, sum2D, header=cubeHead, overwrite=True)

		else:

			if os.path.isfile(saveName):
				pass
			else:
				fits.writeto(saveName, sum2D, header=cubeHead, overwrite=True)



		if returnIntRange:
			return saveName , actualRange

		else:
			return saveName
	def momentFITSGood(self, FITSFile, vRange, mom=0, outFITS=None, overWrite=True, sigma=3, rms=0.5,
					   dv=0.158737644553):
		"""
		:param FITSFile: The fits used to moment, the rms of the fits is 0.5, and the sigma cut is 3, below which data would be deltedted
		:param Vrange: the unite is km/s
		:param mom: usually 0, which is
		:param outFITS:
		:param overWrite:
		:return: No returen, but write the FITS file.
		"""
		# first do a normal moments

		minV = min(vRange)
		maxV = max(vRange)

		vRange = [minV, maxV]

		outMomnetTmp = "outMomnetTmp.fits"  # infact this is used to save the middle fits produced with momentFITS by Miriad

		self.momentFITS(FITSFile, vRange, mom, outFITS=outMomnetTmp, overWrite=True)

		hdu2D = self.downTo2D(outMomnetTmp, outPUT="FITS2D_" + outMomnetTmp, overwrite=True)
		data2D = hdu2D.data
		heade2D = hdu2D.header

		# read FITSFile, and sum the fits mannually

		cubeData, cubeHead = myFITS.readFITS(FITSFile)

		wcs = WCS(cubeHead)

		v0, l0, b0 = wcs.wcs_pix2world(0, 0, 0, 0)

		a, a, indexV0 = wcs.wcs_world2pix(l0, b0, vRange[0] * 1000, 0)

		a, a, indexV1 = wcs.wcs_world2pix(l0, b0, vRange[1] * 1000, 0)

		indexV0, indexV1 = list(map(round, [indexV0, indexV1]))

		indexV0, indexV1 = list(map(int, [indexV0, indexV1]))

		sumData = cubeData[indexV0:indexV1 + 1, :, :]

		sumData[sumData < sigma * rms] = 0

		sum2D = np.sum(sumData, axis=0, dtype=float)
		sum2D = sum2D * dv

		if outFITS == None:
			outFITS = FITSFile[:-5] + "_GoodMomo.fits"

		if overWrite:
			if os.path.isfile(outFITS):
				os.remove(outFITS)
			fits.writeto(outFITS, sum2D, header=heade2D)




	def momentFITS(self, FITSFile, Vrange, mom, outFITS=None, cutEdge=False, overWrite=True):  ##Vrange kms
		"""
		Parameters: FITSFile,Vrange,mom,outPutPath=None,outPutName=None,cutEdge=False

		This method do the moment operature with Miriad

		return the data and header of the moment FITS

		the outputPATH is nessary, because miriad needs a path to run

		The unit of Vrange is kms, and cutEdge is not concerned

		"""

		# If no outPutName or outPUTPATH is provided then no file is going to be saved

		# Split FITSFile

		minVInput = min(Vrange)
		maxVInput = max(Vrange)

		Vrange = [minVInput, maxVInput]

		data, head = self.readFITS(FITSFile)
		head = self.convert4DTo3D(head)

		wcs = WCS(head)

		aa, bb, vMin = wcs.wcs_pix2world(0, 0, 0, 0)
		aa, bb, vMax = wcs.wcs_pix2world(0, 0, data.shape[0] - 1, 0)

		if vMax / 1000. < maxVInput:
			Vrange[1] = vMax / 1000.

		if vMin / 1000. > minVInput:
			Vrange[0] = vMin / 1000.

		processPath, FITSname = os.path.split(FITSFile);

		if processPath == "":
			processPath = "./"

		else:
			processPath = processPath + "/"

		print("================")
		print(processPath, FITSname)

		print("Doing moment {} in the velocity range of {} kms".format(mom, Vrange))

		ReadOutName = "tempoutFITS"

		# if no file name is provided, created one
		# if not outPutName:
		#	outPUTName=FITSFile[0:-5]+"_M{}.fits".format(mom)

		if not outFITS:
			outPath = processPath
			outPutName = FITSname[0:-5] + "_M{}.fits".format(mom)
		else:

			outPath, outPutName = os.path.split(outFITS);

			# print outPath,outPutName,"????????????????????????"

			if outPath == "":
				outPath = "."
			outPath = outPath + "/"

		if not overWrite:
			if os.path.isfile(outPath + outPutName):
				print("No overwriting, skipping.......")
				return self.readFITS(outPath + outPutName)

		# delete this frisrt
		deleteFITS1 = "rm -rf %s" % ReadOutName
		# step 1 read the file
		ReadFITS = "fits in=%s out=%s op=xyin" % (FITSname, ReadOutName)

		# step 2
		##do moment
		# moment in=GCcont out=GCcont.2d mom=-1 region='kms,images(-50,50)'
		momentTempOut = "momenttempout"
		deleteFITS2 = "rm -rf %s" % momentTempOut

		momentString = "moment in=%s out=%s mom=%s region='kms,images(%s,%s)'" % (
			ReadOutName, momentTempOut, mom, Vrange[0], Vrange[1])

		##step 3 output the fits file

		deleteFITS3 = "rm -rf %s" % outPutName

		outPUTFITS = "fits in=%s out=%s op=xyout" % (momentTempOut, outPutName)
		##step3 run commonds

		# goToPath="cd "+outPutPath

		goToPath = "cd " + processPath

		saveScriptPath = "scriptPath=$PWD"
		backToScriptPath = "cd $scriptPath"

		copyFITS = "mv {}{}  {}{}".format(processPath, outPutName, outPath, outPutName)

		if outFITS:
			pass

		self.runShellCommonds(
			[saveScriptPath, goToPath, ReadFITS, momentString, deleteFITS3, outPUTFITS, deleteFITS1, deleteFITS2,
			 backToScriptPath, copyFITS], "./")
		# self.runShellCommonds([saveScriptPath,goToPath,ReadFITS,momentString,deleteFITS3,outPUTFITS,deleteFITS1,deleteFITS2,backToScriptPath,copyFITS],"./")

		return self.readFITS(outPath + outPutName)

	@staticmethod
	def getSpectraByLB(data, dataHeader, l, b):
		"""
		Parameters: data, dataHeader,l,b

		This function is used to get a voxel value from a data cube

		v, km/s
		the unit of returned veloicyt is kms

		return spectral,velocities
		"""
		wcs = WCS(dataHeader)
		xindex, yindex = wcs.all_world2pix(l, b, 0, 0)[0:2]
		xindex = int(np.rint(xindex))
		yindex = int(np.rint(yindex))

		##it is possible that the yindex,and xindex can exceed the boundary of spectrum

		if yindex > data.shape[1] - 1 or xindex > data.shape[2] - 1:
			return None, None
		spectral = data[:, yindex, xindex]
		##just for test
		# print  w.all_world2pix(0, 0, 0,0)
		# print data.shape[0]
		velocityIndex = range(data.shape[0])

		velocities = wcs.all_pix2world(0, 0, velocityIndex, 0)[2] / 1000.

		#
		return spectral, velocities

	@staticmethod
	def getSpectraByIndex(data, dataHeader, indexX, indexY):
		"""
		paramters: data,dataHeader,indexX,indexY
		This function is used to get a voxel value from a data cube

		v, km/s
		"""
		wcs = WCS(dataHeader, naxis=3)

		##it is possible that the yindex,and xindex can exceed the boundary of spectrum

		spectral = data[:, indexY, indexX]
		##just for test
		# print  w.all_world2pix(0, 0, 0,0)
		# print data.shape[0]
		velocityIndex = range(data.shape[0])

		velocities = wcs.all_pix2world(indexX, indexY, velocityIndex, 0)[2] / 1000.

		#
		return spectral, velocities




	@staticmethod
	def downLoadSkyview(survey, centerLB, sizeLB=[0.1, 0.1], resolution=None, overWrite=True,
						savePath="/home/qzyan/WORK/backup/tempskyviewDownload/", saveName=None):
		"""
		all sizes are in degree
		Download IRAS MBM,

		@resoltion of wise 22 0.00038189999999999996

		"""

		print("Download {}, at (l, b)=({}, {}) ----------  sizeL: {} deg; sizeB: {} deg".format(survey, centerLB[0],
																								centerLB[1], sizeLB[0],
																								sizeLB[1]))

		centerL, centerB = centerLB

		sizeL, sizeB = centerLB

		if resolution is None:
			sizePix = 500
		else:
			sizePix = max(sizeLB) / resolution
			sizePix = int(round(sizePix))

		if saveName is None:
			saveName = "{}_{}_{}.fits".format(survey.replace(" ", ""), centerLB[0], centerLB[1])

		print("Saving to  {}{}  ....".format(savePath, saveName))

		if not overWrite and os.path.isfile(savePath + saveName):
			print("File exist, returning....")
			return

		# command="skvbatch_wget file=./{}/{} position='{}, {}' Survey='{}'  Projection='Car' Coordinates='Galactic' Pixels={}".format(savePath,tempName,centerl,centerb,survey,sizePix)
		command = "skvbatch_wget file='{}{}' position='{}, {}' Survey='{}'  Projection='Car' Coordinates='Galactic' Pixels={}".format(
			savePath, saveName, centerL, centerB, survey, sizePix)

		os.system(command)
		# first test a small

		return

		sizePix = 1500

		# survey="IRIS 100"

		# savePath=self.IRASDownloadFolder
		savePath = "MBMFITS"  # IRAS 100 umself, #self.MBMFITSPath

		# saveNameRaw=MBMID+"_IRAS100.fits"

		if survey.endswith('545'):
			saveNameRaw = MBMID + "_planck545.fits"
		if survey.endswith('857'):
			saveNameRaw = MBMID + "_planck857.fits"
		if survey.endswith('100'):
			saveNameRaw = MBMID + "_IRAS100.fits"
		# survey="Planck 857"

		#saveName = saveNameRaw.replace(' ', '\ ')
		if not overWrite:

			outputFile = "./{}/{}".format(savePath, saveName)
			outputFile2 = "./{}/{}".format(savePath, saveNameRaw)

			if os.path.isfile(outputFile2):
				print(outputFile, "exits...doing nothing...")
				return
		tempName = "tempIRAS.fits"  # in case some filenames contains space
		command = "skvbatch_wget file=./{}/{} position='{}, {}' Survey='{}'  Projection='Car' Coordinates='Galactic' Pixels={}".format(
			savePath, tempName, centerl, centerb, survey, sizePix)

		os.system(command)
		copyFile = "mv ./{}/{} ./{}/{} ".format(savePath, tempName, savePath, saveName)
		print(copyFile)
		os.system(copyFile)

	@staticmethod
	def readFITS(fitsFile, ):
		"""
		parameters: fitsFile
		This file will return the data and header of the fits
		"""

		fitsRead = fits.open(fitsFile)

		head = fitsRead[0].header
		try:
			del head["COMMENT"]
			del head["HISTROY"]
		except:
			pass

		header = head.copy()
		# print "Reducing to 3D....if 4D"

		try:
			del header["CRPIX4"]
			del header["CDELT4"]
			del header["CRVAL4"]
			del header["CTYPE4"]
			del header["CROTA4"]

		except:
			pass

		if len(fitsRead[0].data.shape) == 4:
			return fitsRead[0].data[0], header

		else:

			return fitsRead[0].data, header

	def downLoadSurveyByRange(self, Survey, LRange, BRange, Original=True, Pixels=None, size=None, downLoadPath=None):

		"""
		This survey is used to download a survey with the best resolution
		covering the input LBRange

		default size is 0.5 degree,

		modified By QingZeng 08232019
		"""

		processFolder = Survey.replace(" ", "") + "_Mosaic"

		if downLoadPath == None:
			downLoadPath = "./{}/".format(processFolder)

		os.system("mkdir " + downLoadPath)

		centerL, centerB = np.mean(LRange), np.mean(BRange)

		if not Pixels and not size:
			print("No download size is assigned, quit")
			return

		if Original:
			# if the largest pixel resolution is wanted

			resolution = self.detectSurveyResolution(Survey, [centerL, centerB])
			if Pixels and size:
				print("Can't assign size and pixels simultaneously in Original model, quit")
				return

			if size and not Pixels:
				Pixels = size / resolution

		if not Original:

			# in this case, pixels and size must be provided
			if not size:
				size = ""
			if not Pixels:
				Pixels = ""

		# download with pixels and size assigned

		extraPixels = 50

		downLoadSize = size / Pixels * extraPixels + size
		downLoadPixels = Pixels + extraPixels

		downLoadPixels = int(downLoadPixels)  # needs to be integar?

		# estimate pixels

		tilesL = abs(LRange[1] - LRange[0]) / size
		tilesB = abs(BRange[1] - BRange[0]) / size

		minL = min(LRange)
		maxL = max(LRange)

		minB = min(BRange)
		maxB = max(BRange)

		mosaicFITS = []

		print(tilesL, tilesB)

		for i in range(int(tilesL) + 1):
			for j in range(int(tilesB) + 1):

				centerTileL = minL + size * (i + 1)
				centerTileB = minB + size * (j + 1)
				print("Downloading (i,j)=({},{}) fits centered:{} {}".format(i + 1, j + 1, minL + size * (i + 1),
																			 minB + size / 1. * (j + 1)))

				# avoid repeat download
				outputName = downLoadPath + "Mosaic{}{}.fits".format(i + 1, j + 1)
				if os.path.isfile(outputName):
					continue

				# self.getSurvey(Survey,[centerTileL,centerTileB],Pixels=downLoadPixels,size=downLoadSize,outputFITS=outputName)
				self.getSurvey(Survey, [centerTileL, centerTileB], Pixels=downLoadPixels, outputFITS=outputName)

		# print centerTileL,centerTileB
		# if Original:
		# self.getSurvey(Survey,[centerTileL,centerTileB],Pixels=int(sizePixels+50),outputFITS=outputName)
		# else:
		# download by size

		# self.getSurvey(Survey,[centerTileL,centerTileB],Pixels=downLoadPixels,size=downLoadSize,outputFITS=outputName)

		# self.getSurvey(Survey,[centerTileL,centerTileB],size=0.3,outputFITS=outputName)

	# after download, mergethose files with Montage

	def detectSurveyResolution(self, Survey, LB):
		"""

		to know the resolution of the survey
		"""
		# print Survey,LB
		tempPath = "/home/qzyan/astrosoft/ref_qzyan/tempSkyview/"  # a fold to save the resolution of surveys

		saveSurvey = Survey.replace(" ", "")

		outputFITS = tempPath + saveSurvey + "checkRes.fits"

		# print outputFITS, ">>>>>>>>>>>>>>>>"
		if not os.path.isfile(outputFITS):
			self.getSurvey(Survey, LB, outputFITS=outputFITS, Pixels=100)

		data, header = self.readFITS(outputFITS)

		os.system("rm checkRes.fits")

		return abs(header["CDELT1"])  # return degree

	@staticmethod
	def getCloudNameByLB(l, b):

		# if b>=0:

		lStr = str(l)

		bStr = "{:+f}".format(b)

		if '.' in lStr:

			lStrPart1, lStrPart2 = lStr.split('.')

		else:
			lStrPart1 = lStr
			lStrPart2 = '0'

		if '.' in bStr:

			bStrPart1, bStrPart2 = bStr.split('.')
		else:
			bStrPart1 = bStr
			bStrPart2 = '0'

		lStr = lStrPart1 + '.' + lStrPart2[0:1]

		bStr = bStrPart1 + '.' + bStrPart2[0:1]

		lStr = lStr.zfill(5)

		# bStr="{:+.1f}".format(b)
		bStrNumberPart = bStr[1:]
		bStr = bStr[0:1] + bStrNumberPart.zfill(4)

		cName = "G{}{}".format(lStr, bStr)

		return cName

	@staticmethod
	def find_nearestIndex(a, a0):
		"Element in nd array `a` closest to the scalar value `a0`"

		idx = np.abs(a - a0).argmin()
		return idx

	@staticmethod
	def getTBStructure(tb):
		a = Table(tb[0])

		a.remove_row(0)

		return a

	@staticmethod
	def getSurvey(Survey, LB, outputFITS=None, Pixels=None, size=None):
		"""
		parameters: Survey,LB,outputFITS=None, Pixels=500,size=None

		band list is the wise band to be downloaded

		the unit of size is 0.25 degree
		0.25 will keep the original resolution
		size diameter

		"""
		# print "Function works!"

		centerL, centerB = LB

		if not outputFITS:
			outputFITS = Survey.replace(" ", "") + "_" + str(centerL) + "_" + str(centerB) + ".fits"

		if not Pixels and not size:
			# no download paratmers are providided, download with  pixels=500

			command = "skvbatch_wget file={} position='{},{}' Survey='{}'  Coordinates=Galactic  Projection=Car  Pixels={}".format(
				outputFITS, centerL, centerB, Survey, 500)
			print(command)
			os.system(command)
			return

		if Pixels and size:
			command = "skvbatch_wget file={} position='{},{}' Survey='{}'  Coordinates=Galactic  Projection=Car size={} Pixels={}".format(
				outputFITS, centerL, centerB, Survey, size, Pixels)

			# os.system(command)
			return
		if not size:

			command = "skvbatch_wget file={} position='{},{}' Survey='{}'  Coordinates=Galactic  Projection=Car  Pixels={}".format(
				outputFITS, centerL, centerB, Survey, Pixels)
		# print command
		else:
			command = "skvbatch_wget file={} position='{},{}' Survey='{}'  Coordinates=Galactic  Projection=Car size={} ".format(
				outputFITS, centerL, centerB, Survey, size)
		# print command
		# print command
		os.system(command)

	# call("skvbatch file=example1.fits position='+12 34, -10 23'  Survey='Digitized Sky Survey'")
	# call("./skvbatch_wget file=example2.fits position='0,0' Survey='Digitized Sky Survey' Coordinats=Galactic  Projection=Car size=0.5")

	def reverseVelAxis(self, inFITS, outFITS=None):
		"""

		:param inFITS:
		:param outFITS:
		:return:
		"""

		data, head = self.readFITS(inFITS)
		if len(data.shape) == 4:
			head = self.convert4DTo3D(head)

			data = data[0]
		Nz, Ny, Nx = data.shape

		dataReverse = np.flip(data, 0)

		head["CDELT3"] = abs(head["CDELT3"])
		referencePoint = head["CRPIX3"]
		head["CRPIX3"] = Nz - referencePoint + 1

		# needto

		path, baseName = os.path.split(inFITS)
		saveFITS = os.path.join(path, baseName[0:-5] + "_Reverse.fits")

		fits.writeto(saveFITS, dataReverse, header=head, overwrite=True)

		return saveFITS

	@staticmethod
	def checkPath(pathInput):
		"""
		#check if the path exist, if not created it
		:param pathInput:
		:return:
		"""

		if not os.path.isdir(pathInput):
			os.mkdir(  pathInput)

	@staticmethod
	def cropWithCube(inFITS, outFITS=None, Vrange=None, Lrange=None, Brange=None, overWrite=False):

		hdu = fits.open(inFITS)[0]

		data, goodHeader = hdu.data, hdu.header
		# goodHeader["BITPIX"]=
		# datacut=np.float32(datacut) #use 32bit

		try:
			del goodHeader["CRPIX4"]
			del goodHeader["CDELT4"]
			del goodHeader["CRVAL4"]
			del goodHeader["CTYPE4"]
			del goodHeader["CROTA4"]

		except:
			pass

		wmap = WCS(goodHeader, naxis=3)

		if not Vrange and not Lrange and not Brange:
			print("No crop range is provided.")
			return

		# Examine the maximum number for pixels
		if len(data.shape) == 4:
			data = data[0]  # down to 3D

		zSize, ySize, xSize = data.shape

		Xrange = [0, xSize - 1]  # Galactic Longitude  #
		Yrange = [0, ySize - 1]  # Galactic Longitude  #
		Zrange = [0, zSize - 1]  # Galactic Longitude  #

		firstPoint = wmap.wcs_pix2world(0, 0, 0, 0)
		lastPoint = wmap.wcs_pix2world(xSize - 1, ySize - 1, zSize - 1, 0)

		if not Vrange:
			# calculate the range for the
			Zrange = [firstPoint[2], lastPoint[2]]
		else:
			Zrange = np.array(Vrange) * 1000

		if not Lrange:
			Xrange = [firstPoint[0], lastPoint[0]]
		else:
			Xrange = Lrange

		if not Brange:
			Yrange = [firstPoint[1], lastPoint[1]]
		else:
			Yrange = Brange

		# revert Galactic longtitude
		if lastPoint[0] < firstPoint[0]:
			Xrange = [max(Xrange), min(Xrange)]
		# print lastPoint[0],firstPoint[0]

		# print Xrange,Yrange,Zrange
		cutFIRST = wmap.wcs_world2pix(Xrange[0], Yrange[0], Zrange[0], 0)
		cutLAST = wmap.wcs_world2pix(Xrange[1], Yrange[1], Zrange[1], 0)

		cutFIRST = list(map(round, cutFIRST))
		cutLAST = list(map(round, cutLAST))

		cutFIRST = list(map(int, cutFIRST))
		cutLAST = list(map(int, cutLAST))

		cutFIRST[0] = max(0, cutFIRST[0])
		cutFIRST[1] = max(0, cutFIRST[1])
		cutFIRST[2] = max(0, cutFIRST[2])

		cutLAST[0] = min(xSize - 1, cutLAST[0]) + 1
		cutLAST[1] = min(ySize - 1, cutLAST[1]) + 1
		cutLAST[2] = min(zSize - 1, cutLAST[2]) + 1

		#

		cube = SpectralCube.read(inFITS)

		sub_cube = cube[cutFIRST[2]:cutLAST[2], cutFIRST[1]:cutLAST[1], cutFIRST[0]:cutLAST[0]]

		if not outFITS:
			"""
			If no output file Name is provide
			"""
			outFITS = inFITS[:-5] + "_C.fits"

		if not os.path.isfile(outFITS):
			sub_cube.write(outFITS)
		# fits.writeto(outFITS, datacut, header=wmapcut.to_header())

		else:

			if overWrite:
				# delete that file
				os.remove(outFITS)
				sub_cube.write(outFITS)

			# hdu.data=datacut
			# fits.writeto(outFITS, datacut, header=wmapcut.to_header())

			else:
				print("Warring----File ({}) exists and no overwriting!".format(outFITS))
		return outFITS



	@staticmethod
	def cropFITS(inFITS,inputData=None, onlyVel=False, inputHeader=None, outFITS=None, Vrange=None, Lrange=None, Brange=None, overWrite=False, overwrite=False, extend=0,velUnit=None):
		"""
		parameters: inFITS,outFITS=None,Vrange=None,Lrange=None,Brange=None,overWrite=False
		This function is used to create my own function of croping FITS
		Based on Mongate

		In the first version, only Galactic coordinate is supported

		The output fits could be a little bit different as requested

		#no project is concerted in this function

		# the unit of LBV is degree,degree, kms

		#by default we reverse the data

		"""

		if extend>0:
			Lrange=[ min(Lrange)-extend,max(Lrange)+extend ]
			Brange=[ min(Brange)-extend,max(Brange)+extend ]

		# read FITS file
		if inFITS is not None:
			hdu = fits.open(inFITS)[0]

			data, goodHeader = hdu.data, hdu.header

		else:
			data, goodHeader=inputData,inputHeader
		# goodHeader["BITPIX"]=
		# datacut=np.float32(datacut) #use 32bit

		try:
			del goodHeader["CRPIX4"]
			del goodHeader["CDELT4"]
			del goodHeader["CRVAL4"]
			del goodHeader["CTYPE4"]
			del goodHeader["CROTA4"]

		except:
			pass

		wmap = WCS(goodHeader, naxis=3)

		wmap.wcs.bounds_check(False, False)

		if Vrange is None and Lrange is None and Brange is None:
			print("No crop range is provided.")
			return

		# Examine the maximum number for pixels
		if len(data.shape) == 4:
			data = data[0]  # down to 3D

		zSize, ySize, xSize = data.shape

		Xrange = [0, xSize - 1]  # Galactic Longitude  #
		Yrange = [0, ySize - 1]  # Galactic Longitude  #
		Zrange = [0, zSize - 1]  # Galactic Longitude  #

		firstPoint = wmap.wcs_pix2world(0, 0, 0, 0)
		lastPoint = wmap.wcs_pix2world(xSize - 1, ySize - 1, zSize - 1, 0)

		if Vrange is None:
			# calculate the range for the
			Zrange = [firstPoint[2], lastPoint[2]]
		else:  # usually it is not alwasy true that the

			if velUnit is None:

				if abs(goodHeader["CDELT3"]) < 10:  # km /s
					# if "km" in goodHeader["CUNIT3"] or   "Km" in goodHeader["CUNIT3"] or  "KM" in goodHeader["CUNIT3"] :#take it as km/s
					Zrange = np.array(Vrange)
				else:  # m/s
					Zrange = np.array(Vrange) * 1000




			else:  #
				if "km" in velUnit or "Km" in velUnit or "KM" in velUnit:  # take it as km/s
					Zrange = np.array(Vrange)
				else:  # m/s
					Zrange = np.array(Vrange) * 1000

		if Lrange is None:
			Xrange = [firstPoint[0], lastPoint[0]]
		else:
			Xrange = Lrange

		if Brange is None:
			Yrange = [firstPoint[1], lastPoint[1]]
		else:
			Yrange = Brange

		# revert Galactic longtitude
		if lastPoint[0] < firstPoint[0]:
			Xrange = [max(Xrange), min(Xrange)]

		####

		#### this is for 13CO
		if lastPoint[2]   < firstPoint[2]:
			Zrange = [max(Zrange), min(Zrange)]

		# print lastPoint[0],firstPoint[0]
		#print(Zrange,"What the heck is Zrange?")
		# print Xrange,Yrange,Zrange
		cutFIRST = wmap.wcs_world2pix(Xrange[0], Yrange[0], Zrange[0], 0)
		cutLAST = wmap.wcs_world2pix(Xrange[1], Yrange[1], Zrange[1], 0)

		########################################### Added by Qzyan 2020/10/23
		# something wroing with the cutLAST for fits across 180 degress.
		# looks like a difficult issed.
		dl = abs(goodHeader["CDELT1"])
		maximumNx = 360 / dl

		if cutLAST[0] < 0:
			cutLAST[0] = cutLAST[0] + maximumNx

		########################################### Added by Qzyan 2020/10/23

		cutFIRST = list(map(np.round, cutFIRST))
		cutLAST = list(map(np.round, cutLAST))

		cutFIRST = list(map(int, cutFIRST))
		cutLAST = list(map(int, cutLAST))

		cutFIRST[0] = max(0, cutFIRST[0])
		cutFIRST[1] = max(0, cutFIRST[1])
		cutFIRST[2] = max(0, cutFIRST[2])

		cutLAST[0] = min(xSize - 1, cutLAST[0]) + 1
		cutLAST[1] = min(ySize - 1, cutLAST[1]) + 1
		cutLAST[2] = min(zSize - 1, cutLAST[2]) + 1

		# calculate the true pixels according to the input range
		wmapcut = wmap[cutFIRST[2]:cutLAST[2], cutFIRST[1]:cutLAST[1], cutFIRST[0]:cutLAST[0]]
		datacut = data[cutFIRST[2]:cutLAST[2], cutFIRST[1]:cutLAST[1], cutFIRST[0]:cutLAST[0]]
		# datacut=data[1:3,1:5,1:9]

		# hdu = fits.PrimaryHDU(datacut,header=wmapcut)
		datacut = np.float32(datacut)
		if datacut.shape[0]==1:
			datacut= datacut [0]
		#print( np.sqrt(np.nanmean(np.square(datacut))) , "RMS average")


		if not outFITS: #when inFIT is none, the outFITS should be assigned, do not consider this
			"""
			If no output file Name is provide
			"""

			outFITS = inFITS[:-5] + "_C.fits"
		if onlyVel:
			writeHeader = wmapcut.to_header()


			goodHeader["CRPIX3"]=writeHeader["CRPIX3"]
			goodHeader["CDELT3"]=writeHeader["CDELT3"]
			goodHeader["CRVAL3"]=writeHeader["CRVAL3"]
			goodHeader["CTYPE3"]=writeHeader["CTYPE3"]


			writeHeader=goodHeader
		else:
			writeHeader = wmapcut.to_header()

		if not os.path.isfile(outFITS):

			fits.writeto(outFITS, datacut, header= writeHeader )

		else:

			if overWrite or overwrite:
				# delete that file
				os.remove(outFITS)

				# hdu.data=datacut
				fits.writeto(outFITS, datacut, header= writeHeader   )

			else:
				print("Warring----File ({}) exists and no overwriting!".format(outFITS))
		return outFITS

	@staticmethod
	def converto32bit(fitsName, saveFITS=None):
		"""

		:param fitsName:
		:param saveFITS:
		:return:
		"""
		if saveFITS == None:
			fitsName = os.path.split(fitsName)[1]

			saveFITS = fitsName[0:-5] + "32bit.fits"

		data, head = myFITS.readFITS(fitsName)

		fits.writeto(saveFITS, np.float32(data), header=head, overwrite=True)

	@staticmethod
	def convertoms(fitsName, saveFITS=None):
		"""
		concert the thrid axis to m/s
		:param fitsName:
		:param saveFITS:
		:return:
		"""
		if saveFITS == None:
			fitsName = os.path.split(fitsName)[1]

			saveFITS = fitsName[0:-5] + "ms.fits"

		data, head = myFITS.readFITS(fitsName)

		if "km" in head["CUNIT3"] or "Km" in head["CUNIT3"] or "KM" in head["CUNIT3"]:
			head["CUNIT3"] = "m/s"
			head["CRVAL3"] = head["CRVAL3"] * 1000
			head["CDELT3"] = head["CDELT3"] * 1000

			fits.writeto(saveFITS, data, header=head, overwrite=True)

		else:

			return fitsName

	@staticmethod
	def creatPPVHeader(fitsName, saveFITS=None):
		"""
		produce the LV header for a fitsName, this function used in dendrogram
		:param fitsName:
		:param saveFITS:
		:return:
		"""

		if saveFITS == None:
			fitsName = os.path.split(fitsName)[1]

			saveFITS = fitsName[0:-5] + "LVHeader.fits"

		CO12HDU = fits.open(fitsName)[0]
		data, head = myFITS.readFITS(fitsName)

		#
		wcs = WCS(head)

		Nz, Ny, Nx = data.shape
		beginP = [0, (Ny - 1.) / 2.]

		endP = [(Nx), (Ny - 1.) / 2.]
		# get pv diagrame
		widthPix = 2


		endpoints = [beginP, endP]
		xy = ppvPath(endpoints, width=widthPix)

		pv = extract_pv_slice(CO12HDU, xy)

		os.system("rm " + saveFITS)

		pv.writeto(saveFITS)

		if 1:  # modify the first

			pvData, pvHead = myFITS.readFITS(saveFITS)

			pvHead["CDELT1"] = head["CDELT1"]
			pvHead["CRPIX1"] = head["CRPIX1"]
			pvHead["NAXIS1"] = head["NAXIS1"]

			pvHead["CRVAL1"] = head["CRVAL1"]

			os.system("rm " + saveFITS)

			fits.writeto(saveFITS, pvData, header=pvHead, overwrite=True)

	@staticmethod
	def getRMS(Array):

		"""
		Return the RMS of Array
		"""

		Array = np.array(Array)

		return np.sqrt(np.mean(np.square(Array)))

	def getSpecRMS(self,spectrum):
		"""

		:param spectrum:
		:return:
		"""

		negativeValues = spectrum[spectrum<=0]

		return self.getRMS(negativeValues)


	@staticmethod
	def getCOFITSNoise(FITSName):

		"""
		Return the nose RMS of FITS file
		"""

		# read fits

		fitsRead = fits.open(FITSName)

		data = fitsRead[0].data

		negative = data[data < 0]

		return np.sqrt(np.mean(np.square(negative)))

	# use the negative values at

	# p=np.abs(negative)

	# variance=np.var(p,ddof=1)

	# ab=1-2./np.pi

	# return np.sqrt( variance/ab  )


	def getLBRangeByFITS(self, fitsName):
		"""
		get the l,b range of fits, by calculating the first and last point
		:param self:
		:param fitsName:
		:return:
		"""
		data,header = self.readFITS(fitsName)
		if len( data.shape)==3:


			zSize,ySize,xSize=data.shape

		else:
			ySize, xSize = data.shape

		wmap =  WCS(header, naxis=2) #first 2
		firstPoint = wmap.wcs_pix2world(0, 0, 0)
		lastPoint = wmap.wcs_pix2world(xSize - 1, ySize - 1, 0)

		lRange=[lastPoint[0],firstPoint[0]]

		bRange=[firstPoint[1], lastPoint[1] ]

		return lRange,bRange




	@staticmethod
	def cropFITS2D(inFITS, outFITS=None, Lrange=None, Brange=None, overWrite=False,extend=0,):
		"""
		parameters: inFITS,outFITS=None,Vrange=None,Lrange=None,Brange=None,overWrite=False
		Thiss function is used to create my own function of croping FITS
		Based on Mongate

		In the first version, only Galactic coordinate is supported

		The output fits could be a little bit different as requested

		#no project is concerted in this function

		# the unit of LBV is degree,degree, kms
		"""

		if extend>0:
			Lrange=[ min(Lrange)-extend,max(Lrange)+extend ]
			Brange=[ min(Brange)-extend,max(Brange)+extend ]

		if not os.path.isfile(inFITS):
			print("file dose not exist ",inFITS)
			return

		# read FITS file

		hdu = fits.open(inFITS)[0]

		header, data = hdu.header, hdu.data

		wmap = WCS(header, naxis=2)
		sizeData = data.shape
		if len(sizeData) == 3:
			data = data[0]
		if len(sizeData) == 4:
			data = data[0]
			data = data[0]

		if  Lrange is None and Brange is None:
			print("No crop range is provided.")
			return

		# Examine the maximum number for pixels

		ySize, xSize = data.shape

		"Dose this work?"

		Xrange = [0, xSize - 1]  # Galactic Longitude  #
		Yrange = [0, ySize - 1]  # Galactic Longitude  #

		firstPoint = wmap.wcs_pix2world(0, 0, 0)
		lastPoint = wmap.wcs_pix2world(xSize - 1, ySize - 1, 0)

		if  Lrange is   None:
			Xrange = [firstPoint[0], lastPoint[0]]

		else:
			Xrange = Lrange
		if  Brange is    None :
			Yrange = [firstPoint[1], lastPoint[1]]
		else:
			Yrange = Brange

		# revert Galactic longtitude
		# if lastPoint[0]<firstPoint[0]:
		Xrange = [max(Xrange), min(Xrange)]
		Yrange = [min(Yrange), max(Yrange)]

		# print lastPoint[0],firstPoint[0]

		# print Xrange,Yrange,Zrange
		cutFIRST = wmap.wcs_world2pix(Xrange[0], Yrange[0], 0)
		cutLAST = wmap.wcs_world2pix(Xrange[1], Yrange[1], 0)

		# WWWWWWWWWWWWWWWWW

		cutFIRST = list(map(np.round, cutFIRST))
		cutLAST = list(map(np.round, cutLAST))

		cutFIRST = list(map(int, cutFIRST))
		cutLAST = list(map(int, cutLAST))

		cutFIRST[0] = max(0, cutFIRST[0])
		cutFIRST[1] = max(0, cutFIRST[1])

		cutLAST[0] = min(xSize - 1, cutLAST[0]) + 1
		cutLAST[1] = min(ySize - 1, cutLAST[1]) + 1

		# print cutFIRST,"first"
		# print cutLAST,"last"

		# calculate the true pixels according to the input range
		wmapcut = wmap[cutFIRST[1]:cutLAST[1], cutFIRST[0]:cutLAST[0]]
		datacut = data[cutFIRST[1]:cutLAST[1], cutFIRST[0]:cutLAST[0]]
		# datacut=data[1:3,1:5,1:9]

		# hdu = fits.PrimaryHDU(datacut,header=wmapcut)

		if   outFITS is  None :
			"""
			If no output file Name is provide
			"""
			outFITS = inFITS[:-5] + "_C.fits"


		writeHeader=  wmapcut.to_header()
		# BMIN, and beammajor is used for the AG map and CO match,
		# larger beam sized means low matches



		if "BMAJ" in header.keys():
			writeHeader["BMAJ"] = header["BMAJ"]

		if "BMIN" in header.keys():
			writeHeader["BMIN"] = header["BMIN"]

		#


		if not os.path.isfile(outFITS):

			fits.writeto(outFITS, datacut, header= writeHeader )

		else:

			if overWrite:

				fits.writeto(outFITS, datacut, header= writeHeader ,overwrite=True )

			else:
				print("Warring----File ({}) exists and no overwriting!".format(outFITS))

		return outFITS

	@staticmethod
	###############Static functions#######################
	def runShellCommonds(commondList, processPath):

		"""
		parameters: commondList,processPath
		run shell commonds
		"""
		##write each commondList into a file and run them and then delete this file

		tempSHfile = processPath + "runcommondlistTemp.sh"

		f = open(tempSHfile, 'w')
		for eachLine in commondList:
			f.write(eachLine + "\n")
		f.close()

		os.system("bash  %s" % (tempSHfile) + "   >/dev/null 2>&1")  # the end string supress the output of os.system
		# delete file

		os.system("rm -rf " + tempSHfile)

	# self.l=l_input
	# self.b=b_input

	def getComplex(self,amp,angle):
		"""
		The angle is in degree
		:param amp:
		:param angle:
		:return:
		"""
		rad=np.deg2rad(angle)

		return amp*np.cos(rad) +1j*np.sin(rad )*amp


	def getAngle(self,complexN):
		"""

		:return:
		"""

		return np.rad2deg(np.angle(complexN))
		#return np.rad2deg(np.arctan(complexN.imag / complexN.real))

	@staticmethod
	def TB1InTB2(TB1File, TB2File, testCol="_idx"):

		"""
		TB2 is the larger one
		:param TB1File:
		:param TB2File:
		:return:
		"""

		TB1 = Table.read(TB1File)
		TB2 = Table.read(TB2File)

		index1 = TB1[testCol]
		index2 = TB2[testCol]

		selectCriteria = np.in1d(index2, index1)  #

		newTB1 = TB2[selectCriteria]
		newTB1.write("reduce_" + TB2File, overwrite=True)

	@staticmethod
	def drawBV(fitsName, saveFITS=None, RMS=0.5, cutLevel=3.):

		if saveFITS == None:
			saveFITS = fitsName[:-5] + "_BV.fits"

		CO12HDU = fits.open(fitsName)[0]
		data, head = CO12HDU.data, CO12HDU.header

		wcs = WCS(head)

		Nz, Ny, Nx = data.shape
		# beginP = [0, (Ny - 1.) / 2.]

		# endP = [(Nx), (Ny - 1.) / 2.]
		beginP = [(Nx - 1.) / 2., 0]

		endP = [(Nx - 1.) / 2., Ny]
		widthPix = 2


		endpoints = [beginP, endP]
		xy = ppvPath(endpoints, width=widthPix)

		pv = extract_pv_slice(CO12HDU, xy)

		# os.system("rm " + saveFITS)

		pv.writeto(saveFITS, overwrite=True)

		if 1:  # modify the first

			pvData, pvHead = myFITS.readFITS(saveFITS)

			pvHead["CDELT1"] = head["CDELT2"]
			pvHead["CRPIX1"] = head["CRPIX2"]
			pvHead["NAXIS1"] = head["NAXIS2"]
			pvHead["CRVAL1"] = head["CRVAL2"]

		# data[data<cutLevel*RMS ]=0 #by default, we remove those emissions less than 3 sgima

		# resolution
		res = 30. / 3600.
		PVData2D = np.sum(data, axis=2, dtype=float) * res

		if PVData2D.shape == pvData.shape:
			# .....
			# os.system("rm " +saveFITS )

			fits.writeto(saveFITS, PVData2D, header=pvHead, overwrite=True)
		else:

			print("The shape of pvdata with manual integration is unequal!")

	@staticmethod
	def addSuffix(fileName, suffix):
		"""
		add a suffix to the fileName
		:param fileName:
		:return:
		"""
		pathStr, ext = os.path.splitext(fileName)

		return pathStr + suffix + ext



	@staticmethod
	def \
			getLVFITS(fitsName, saveFITS=None, redo=True, maskFITS=None):

		if saveFITS == None:
			saveFITS = fitsName[:-5] + "_LV.fits"

		if maskFITS is not None:
			pathStr, ext = os.path.splitext(saveFITS)
			maskStr = "_mask"
			saveFITS = pathStr + maskStr + ext

		if not redo:
			return saveFITS

		CO12HDU = fits.open(fitsName)[0]
		data, head = CO12HDU.data, CO12HDU.header

		if maskFITS is not None:  # mask the COFITS

			CO12HDUMask = fits.open(maskFITS)[0]
			dataMask = CO12HDUMask.data
			data[dataMask == 0] = 0




			del dataMask
			gc.collect()

		wcs = WCS(head)

		Nz, Ny, Nx = data.shape
		beginP = [0, (Ny - 1.) / 2.]

		endP = [(Nx), (Ny - 1.) / 2.]

		widthPix = 2


		endpoints = [beginP, endP]
		xy = ppvPath(endpoints, width=widthPix)

		pv = extract_pv_slice(CO12HDU, xy)

		# os.system("rm " +saveFITS )

		pv.writeto(saveFITS, overwrite=True)

		if 1:  # modify the first
			pvData, pvHead = myFITS.readFITS(saveFITS)

			pvHead["CDELT1"] = head["CDELT1"]
			pvHead["CRPIX1"] = head["CRPIX1"]
			pvHead["NAXIS1"] = head["NAXIS1"]
			pvHead["CRVAL1"] = head["CRVAL1"]

			pvHead["CDELT2"] = pvHead["CDELT2"] / 1000.

			pvHead["CRVAL2"] = pvHead["CRVAL2"] / 1000.

			pvHead["CUNIT2"] = 'm/s'

		# data[data<cutLevel*RMS ]=0 #by default, we remove those emissions less than 3 sgima

		PVData2D = np.mean(data, axis=1, dtype=float)  # *res

		if PVData2D.shape == pvData.shape:

			fits.writeto(saveFITS, PVData2D, header=pvHead, overwrite=True)
		else:

			print("The shape of pvdata with manual integration is unequal!")

		return saveFITS




	def selectTBByColRange(self, TB, colName, minV=None, maxV=None):
		"""
		Select colnames to form a new table
		:param TB:
		:param colList:
		:return:
		"""

		range = [minV, maxV]
		if range[0] is None and range[1] is None:
			return TB

		if range[0] is None and range[1] is not None:  # upper cut

			selectCriteria = TB[colName] <= range[1]

			return TB[selectCriteria]

		if range[0] is not None and range[1] is None:  # lower cut

			selectCriteria = TB[colName] >= range[0]

			return TB[selectCriteria]

		if range[0] is not None and range[1] is not None:  # both lower and upper cut
			selectCriteria1 = TB[colName] >= range[0]
			selectCriteria2 = TB[colName] <= range[1]
			selectCriteria = np.logical_and(selectCriteria1, selectCriteria2)
			return TB[selectCriteria]

	def showCloudPositions(self, cloudTB):
		"""
		Only used for cloud tables
		:param cloudTB:
		:return:
		"""

		newTBColList = [cloudTB["peakL"], cloudTB["peakB"], cloudTB["peakV"], cloudTB["allChannel"],
						cloudTB["peakChannel"], cloudTB["conseCol"]]

		print(Table(newTBColList))

	def selectTBByCols(self, TB, colNameList):
		"""
		Select colnames to form a new table
		:param TB:
		:param colList:
		:return:
		"""

		colList = []

		for eachName in colNameList:
			colList.append(TB[eachName])

		return Table(colList)

	def writeTreeStructure(self, dendro, saveName):

		f = open(saveName, 'w')

		# for eachC in self.dendroData:
		for eachC in dendro:

			parentID = -1

			p = eachC.parent

			if p != None:
				parentID = p.idx

			fileRow = "{} {}".format(eachC.idx, parentID)
			f.write(fileRow + " \n")

		f.close()

	def mimic4D(self, fitsCube, saveName, templateFITS="/media/qzyan/maclinux/Data/MWISP/G210Deep/2180-005U.fits"):
		"""

		:param self:
		:param fitsCube:
		:param templateFITS:
		:param saveName:
		:return:
		"""
		######
		data, head = self.readFITS(fitsCube)

		fitsRead = fits.open(templateFITS)

		headModel = fitsRead[0].header

		newData = np.asarray([data])
		print(newData.shape)

		####
		newHead = head
		# newHead["NAXIS4"] =  headModel["NAXIS4"]
		newHead["CTYPE4"] = headModel["CTYPE4"]
		newHead["CRVAL4"] = headModel["CRVAL4"]
		newHead["CDELT4"] = headModel["CDELT4"]
		newHead["CRPIX4"] = headModel["CRPIX4"]
		newHead["CROTA4"] = headModel["CROTA4"]

		fits.writeto(saveName, newData, header=newHead, overwrite=True)

	def nanTo1000(self, fitsCube):
		"""

		:param self:
		:param fitsCube:
		:return:
		"""
		data, head = self.readFITS(fitsCube)

		data[np.isnan(data)] = -1000

		fits.writeto(fitsCube, data, header=head, overwrite=True)

	def nanTo0(self, fitsCube):
		"""

		:param self:
		:param fitsCube:
		:return:
		"""
		data, head = self.readFITS(fitsCube)

		data[np.isnan(data)] = 0

		fits.writeto(fitsCube, data, header=head, overwrite=True)

	def getLBVRange(self, fitsFile):
		"""

		:param data:
		:param head:
		:return:
		"""

		####

		data, head = self.readFITS(fitsFile)

		if len(data.shape) == 4:
			data = data[0]

		wcsCO = WCS(head, naxis=3)
		Nz, Ny, Nx = data.shape

		l0, b0, v0 = wcsCO.wcs_pix2world(0, 0, 0, 0)
		l1, b1, v1 = wcsCO.wcs_pix2world(Nx - 1, Ny - 1, Nz - 1, 0)

		return [l1, l0], [b0, b1], [v0 / 1000., v1 / 1000.]

	def getDefaultRMSFITSname(self, fitsCube,subfix=""):
		"""

		:param coFITS:
		:return:
		"""
		dataPath, fitsNameBase = os.path.split(fitsCube)
		saveName = os.path.join(dataPath, fitsNameBase[0:-5] + "_rms{}.fits".format(subfix))
		return saveName

	def getRMSFITS(self, fitsCube, saveName=None, returnRMSValue=False, returnData=False,subfix=""):
		pass

		data, head = self.readFITS(fitsCube)
		if len(data.shape) == 4 and data.shape[0] == 1:
			data = data[0]

		# remove positive values
		data[data >= 0] = np.nan
		data[data <-500] = np.nan #remove bad marke

		##
		dataWithPositive = np.vstack([data, -data])

		stdFITS = np.nanstd(dataWithPositive, axis=0, ddof=1)
		# stdFITS=stdFITS/np.sqrt( 1-2./np.pi )

		# stdFITSByMean=-np.nanmean(data, axis=0 )
		# stdFITSByMean=stdFITSByMean*np.sqrt(np.pi)/np.sqrt(2)

		if returnRMSValue:
			return np.nanmean(stdFITS)

		if returnData:
			return stdFITS

		if saveName is None:
			saveName = self.getDefaultRMSFITSname(fitsCube,subfix=subfix)
		fits.writeto(saveName, stdFITS, header=head, overwrite=True)
		return saveName

	##########################################################################################
	def selectTBFormal(self, TB, cutOff=2, pixN=16, minDelta=3, hasBeam=True, minChannel=3):
		"""
		# This is the most strict critera to select, first by pixN, second by miNDelta, which only affect peak, the peak is calculated by cuOff+minDelta,
		# minChannel and has Beam would also be done
		:param TB:
		:param areaPix:
		:param conChannel:
		:return:
		"""

		# first, check cuOff, to prevent wrong cutOff

		# first voxel
		filterTB = TB[TB["pixN"] >= pixN]

		# second peak
		filterTB = filterTB[filterTB["peak"] >= (minDelta + cutOff) * self.rmsCO12]

		# third by beam,
		if hasBeam:  # this is
			filterTB = filterTB[filterTB["has22"] >= 0.5]

		# select by minCHannel
		filterTB = filterTB[filterTB["allChannel"] >= minChannel]

		# reamove edged
		return self.removeWrongEdges(filterTB)

	def removeWrongEdges(self, TB):

		if TB == None:
			return None
		processTB = TB.copy()

		# remove cloudsThat touches the noise edge of the fits

		# part1= processTB[ np.logical_and( processTB["x_cen"]>=2815 ,processTB["y_cen"]>= 1003  )   ] #1003, 3.25

		# part2= processTB[ np.logical_and( processTB["x_cen"]<= 55 ,processTB["y_cen"]>= 1063  )   ] #1003, 3.25

		if "peak" in TB.colnames:  # for db scan table

			# part1= processTB[ np.logical_or( processTB["x_cen"]>26.25 ,processTB["y_cen"] < 3.25  )   ] #1003, 3.25
			part1 = processTB[
				np.logical_or(processTB["x_cen"] > 26.25, processTB["y_cen"] < 3.25)]  # 1003, 3.25

			# part2= part1[ np.logical_or( part1["x_cen"]<49.25 ,part1["y_cen"]<  3.75 )   ] #1003, 3.25
			part2 = part1[np.logical_or(part1["x_cen"] < 49.25, part1["y_cen"] < 3.75)]  # 1003, 3.25

			return part2
		else:  # dendrogram tb

			part1 = processTB[np.logical_or(processTB["x_cen"] < 2815, processTB["y_cen"] < 1003)]  # 1003, 3.25

			part2 = part1[np.logical_or(part1["x_cen"] > 55, part1["y_cen"] < 1063)]  # 1003, 3.25

			return part2

	def mergeByVaxis(self, fits1, fits2, outPut="mergedCube.fits"):
		"""
		#takes two fits files, and merge them together, to see if the SASMA can process this large data
		we merge the local and the perseus arm file,

		:param fits1:
		:param fits2:
		:param outPut:
		:return:
		"""

		# find the fits, that has the lowerest velocity, and append is to the

		data1, head1 = self.readFITS(fits1)

		data2, head2 = self.readFITS(fits2)

		spec1, vaxis1 = self.getSpectraByIndex(data1, head1, 0, 0)
		spec2, vaxis2 = self.getSpectraByIndex(data2, head2, 0, 0)

		if vaxis1[0] < vaxis2[0]:

			lowData, lowHead = data1, head1
			highData, highHead = data2, head2

			lowSpec, lowVaxis = spec1, vaxis1
			highSpec, highVaxis = spec2, vaxis2

		else:

			lowData, lowHead = data2, head2
			highData, highHead = data1, head1

			lowSpec, lowVaxis = spec2, vaxis2
			highSpec, highVaxis = spec1, vaxis1

		# process low and high

		maxVlow = lowVaxis[-1]

		minVHigh = highVaxis[0]

		# find and middle position

		middleV = (maxVlow + minVHigh) / 2.

		mergeIndexLow = self.find_nearestIndex(lowVaxis, middleV)

		mergeV = lowVaxis[mergeIndexLow]

		part1data = lowData[0:mergeIndexLow]

		mergeIndexHigh = self.find_nearestIndex(highVaxis, mergeV)

		part2data = highData[mergeIndexHigh:]

		if highVaxis[mergeIndexHigh] != mergeV:
			print("Two fits has different cooridnate at velocity, cannot do direct merge, exist... ")
			return

		print("Merging at {:.3f} km/s".format(mergeV))

		mergeData = np.vstack([part1data, part2data])

		fits.writeto(outPut, mergeData, header=lowHead, overwrite=True)
		return outPut

	def calRMS(self, arrayA):
		"""
		:param self:
		:param arrayA:
		:return:
		"""

		return np.sqrt(np.mean(np.square(arrayA)))

	def Univ_Rc(self, r, a2=0.96, a3=1.62, R0=8.15):
		'''
			Rotation curve moldel
			r : kpc
			:type r: object
			'''

		lambd = (a3 / 1.5) ** 5
		R0pt = a2 * R0
		rho = r / R0pt
		log_lam = np.log10(lambd)
		term1 = 200.0 * lambd ** 0.41

		top = 0.75 * np.exp(-0.4 * lambd)
		bot = 0.47 + 2.25 * lambd ** 0.4
		term2 = np.sqrt(0.80 + 0.49 * log_lam + (top / bot))

		top = 1.97 * rho ** 1.22
		bot = (rho ** 2 + 0.61) ** 1.43
		term3 = (0.72 + 0.44 * log_lam) * (top / bot)

		top = rho ** 2
		bot = rho ** 2 + 2.25 * lambd ** 0.4
		term4 = 1.6 * np.exp(-0.4 * lambd) * (top / bot)

		Tr = (term1 / term2) * np.sqrt(term3 + term4)
		return Tr

	@staticmethod
	def filterByRange(TB, colName, vRange):
		"""

		filter the table by the name and vRange

		"""

		if vRange is None:
			return TB

		if len(vRange)  == 2 and vRange[0] is None and vRange[1] is None:
			return TB



		if len(vRange) == 1:
			doTB = TB.copy()

 
			returnTB = doTB[doTB[colName] == vRange[0]]
			return returnTB

		if vRange[0] is None:

			doTB = TB.copy()

			#doTB.add_index(colName)

			returnTB = doTB[doTB[colName] <= vRange[1]]

			#returnTB = doTB.loc[colName, :vRange[1]]

			#TestTB = Table()

			#for eachCol in returnTB.colnames:
				#aaa = Column(returnTB[eachCol], name=eachCol)

				#TestTB.add_column(aaa)
			# gaiaOffCloudStars.remove_column(self.coint)
			return returnTB.copy()

		if vRange[1] is  None:

			doTB = TB.copy()


			#returnTB = doTB.loc[colName, vRange[0]:]


			returnTB = doTB[doTB[colName] >= vRange[0]]

			return returnTB.copy()

		# add inDex
		doTB = TB.copy()

		doTB.add_index(colName)

		returnTB = doTB.loc[colName, min(vRange):max(vRange)]

		TestTB = Table()

		for eachCol in returnTB.colnames:
			aaa = Column(returnTB[eachCol], name=eachCol)

			TestTB.add_column(aaa)
		# gaiaOffCloudStars.remove_column(self.coint)
		return TestTB

	def getLBrangeWithMask(self, maskFITS, extendByDeg=None):
		"""
		"""

		# get the least region box, that contains the mask fits

		data, head = myFITS.readFITS(maskFITS)

		axisX = np.sum(data, axis=0)
		axisY = np.sum(data, axis=1)

		nonZerosX = np.nonzero(axisX)[0]

		firstX = nonZerosX[0]
		secondX = nonZerosX[-1]

		nonZerosY = np.nonzero(axisY)[0]

		firstY = nonZerosY[0]
		secondY = nonZerosY[-1]

		tempWCS = WCS(head, naxis=(1, 2))

		endL, startB = tempWCS.wcs_pix2world(firstX, firstY, 0)

		startL, endB = tempWCS.wcs_pix2world(secondX, secondY, 0)

		lRange = [min([endL, startL]), max([endL, startL])]

		bRange = [min([startB, endB]), max([startB, endB])]

		if extendByDeg is not None:
			return self.expandLBRangeByDeg(lRange, bRange, data, tempWCS, extendByDeg, extendByDeg)

		return lRange, bRange

	def expandLBRangeByDeg(self, lRange, bRange, backData, backWCS, lExpand=0.5, bExpand=0.25):
		"""
		###use absolute expand, to controle the size of box,
		:param lRange:
		:param bRange:
		:param backData:
		:param backWCS:
		:param ratioEx:
		:return:
		"""

		sizeY, sizeX = backData.shape
		maxL, minB = backWCS.wcs_pix2world(0, 0, 0)
		minL, maxB = backWCS.wcs_pix2world(sizeX - 1, sizeY - 1, 0)

		newSmallL = max(min(lRange) - lExpand, minL)

		newSmallB = max(min(bRange) - bExpand, minB)

		newLargeL = min(max(lRange) + lExpand, maxL)

		newLargeB = min(max(bRange) + bExpand, maxB)

		return [newSmallL, newLargeL], [newSmallB, newLargeB]




	def getLatexlName(self, cloudName):

		"""

		:param cloudName:
		:return:

		"""
		if "+" in cloudName:

			return r"{}{}{}".format(cloudName.split("+")[0], "$+$", cloudName.split("+")[1])

		else:
			return r"{}{}{}".format(cloudName.split("-")[0], "$-$", cloudName.split("-")[1])



	def reviseHeadForAGC(self, fitsName, saveName=None):
		"""
		revise header to make the fits normal for Galatci anticenter region
		:param fitsName:
		:return:
		"""

		data, head = self.readFITS(fitsName)  ####

		previousCRVAL1 = head["CRVAL1"]
		previousCRPIX1 = head["CRPIX1"]

		head["CRVAL1"] = 180

		head["CRPIX1"] = previousCRPIX1 + (180 - previousCRVAL1) / abs(head["CDELT1"])
		if saveName is None:
			pathName, bashName = os.path.split(fitsName)
			saveName = os.path.join(pathName, bashName[0:-5] + "_AGC.fits")
		fits.writeto(saveName, data, header=head, overwrite=True)
		return saveName
	def produceNoiseFITS(self, channelN, rmsFITS=None, inputRmsData=None):
		"""
		based on the rmsFITS, produce a spectral with channelN,  and add this cube to smooth fits to smooth fits
		smooth fits, needto calculate rms fits, better use outArmFITS,
		:param rmsFITS:
		:param channelN:
		:return:
		"""
		if inputRmsData is None:
			dataRMS, headRMS = self.readFITS(rmsFITS)

		else:
			dataRMS = inputRmsData

		np.random.seed()

		Ny, Nx = dataRMS.shape
		Nz = channelN

		dataNoise = np.zeros((Nz, Ny, Nx), dtype=np.float)

		for j in range(Ny):
			for i in range(Nx):

				if np.isnan(dataRMS[j, i]):
					continue

				dataNoise[:, j, i] = np.random.normal(0, dataRMS[j, i], Nz)

		return dataNoise

	def maskBadChannel(self, DBSCANobj, outPutFITS=None, beginV=33.9699, endV=34.4461):
		"""
		this function is to remove a badchannel around 34 km/s, either we remove those clouds ,,manuually

		The method is to set all voxels between beginChannel and endChannel to zeor, if the correpsonding spectra has no emission in beginChannel and endChannel

		:param rawCOFITS:
		:param labelFITSClean:
		:return:
		"""

		rawCOFITS = DBSCANobj.rawCOFITS
		labelFITSClean = DBSCANobj.cleanLabelFITSName
		rmsFITS = DBSCANobj.rmsFITS

		if outPutFITS is None:
			pathRaw, baseRaw = os.path.split(rawCOFITS)
			outPutFITS = os.path.join(pathRaw, baseRaw[0:-5] + "_RM34.fits")

		dataLabel, headLabel = self.readFITS(labelFITSClean)
		dataRaw, headRaw = self.readFITS(rawCOFITS)
		dataRMS, headRMS = self.readFITS(rmsFITS)

		wcsCO = WCS(headLabel, naxis=3)
		_, _, beginChannel = wcsCO.wcs_world2pix(0, 0, beginV * 1000, 0)

		_, _, endChannel = wcsCO.wcs_world2pix(0, 0, endV * 1000, 0)

		beginChannel = np.round(beginChannel)
		beginChannel = int(beginChannel)

		endChannel = np.round(endChannel)

		endChannel = int(endChannel)

		###first get the begin change and end channel of 34 km/s

		# if beginChannel is None or endChannel is None:  # the unit should be index

		# begionChannel, and endChannel are all bad channels

		if len(dataRMS.shape) == 3:
			dataRMS = dataRMS[0]

		labelSub = dataLabel[beginChannel: endChannel + 1]
		dataSub = dataRaw[beginChannel: endChannel + 1]

		# do this elegently
		# identifing regions to be replaced

		# produce noise according to dataRMS
		Nz, Ny, Nx = dataSub.shape
		noiseSub = self.produceNoiseFITS(Nz, inputRmsData=dataRMS)

		maskChannel = dataLabel[beginChannel - 1] + dataLabel[endChannel + 1]
		maskChannel = np.asarray([maskChannel])
		maskSub = np.repeat(maskChannel, Nz, axis=0)

		dataSub[maskSub == 0] = noiseSub[maskSub == 0]

		dataRaw[beginChannel: endChannel + 1] = dataSub

		fits.writeto(outPutFITS, dataRaw, header=headRaw, overwrite=True)

	###################

	def drawFITS(self, fitsFile, saveName=None, savePath='./', colorCmap="jet", reverseX=False,rms=None ):
		"""

		:param fitsFile:
		:param savePath:
		:return:
		"""

		data, header = self.readFITS(fitsFile)

		if len(data.shape) == 3:
			data = data[0]  # []
		if rms is None:
			rms = self.getRMS(data)

		wcsCO = WCS(header, naxis=2)
		fig = plt.figure(1, figsize=(8, 4.5))

		fontsize = 18

		rc('text', usetex=True)
		rc('font', **{'family': 'sans-serif', 'size': fontsize, 'serif': ['Helvetica']})
		mpl.rcParams['text.latex.preamble'] = r'\usepackage{tgheros} \usepackage{sansmath} \sansmath'

		ax = plt.subplot(projection=wcsCO)
		ax.imshow(data, origin='lower', cmap=colorCmap, vmin=0.5 * rms, vmax=10 * rms, interpolation='none')
		if saveName is None:
			fileBase = os.path.basename(fitsFile)
			saveName = os.path.join(savePath, fileBase[0:-5] + '.png')

		if reverseX:
			ax.invert_xaxis()

		plt.axis('off')

		plt.savefig(saveName, bbox_inches="tight", dpi=600, pad_inches=0)


	def diffFITS(self,fits1,fits2,saveName="diff_2_1.fits"):
		"""
		subtract  the data is data2 #### for server using
		:param fits1:
		:param fits2:
		:param saveName:
		:return:
		"""
		data1,head1=self.readFITS(fits1)
		data2,head2= self.readFITS(fits2)
		diffData=data2-data1

		fits.writeto(saveName,diffData, header = head1 , overwrite=True )



	def diffChannel(self,fitsName, saveName=None):
		"""
		subtract  the data is data2 #### for server using
		:param fits1:
		:param fits2:
		:param saveName:
		:return:
		"""

		if saveName is None:
			saveName = self.addSuffix(fitsName, "_diffChannel" )
			saveName = os.path.split(saveName)[1]

			#### save to the current path



		####
		rawData, rawHead = self.readFITS( fitsName )
		Nz, Ny, Nx = rawData.shape


		extendArray = np.zeros((Nz + 1, Ny, Nx), dtype=float)


		extendArray[1:] = rawData
		diffArray = extendArray[1:] - extendArray[0:-1]

		fits.writeto( saveName , diffArray, header = rawHead )

		return saveName

	def getVFromHead(self,head):

		return abs(head["CDELT3"]/1000.)
	def ZZZ(self):
		# mark the end of the file
		pass

