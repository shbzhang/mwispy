#!/usr/bin/env python
import sys
#from gaiaDis import GAIADIS
import os
import argparse
from astropy.io import fits
import gc
import glob
import numpy as np

from astropy.wcs import WCS

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib.offsetbox import AnchoredText

import matplotlib
matplotlib.use('Agg')



"""
Copyright declare 
Copyright (c) 2023 Yan, Qing-Zeng for MWISP, all right reserved.

"""

"""
Author: Qing-Zeng Yan
V1.1 : the axis range is incorrect, fixed by inserting the maximum and minimum values ..
V3.0 (Revised at 2023/11/13)
# fix the channel number display malfunction
V4.0 The compatable problem Qinghai station
"""


class drawCube():
    wcs = None
    wcs2D = None ### for  drawing
    processPath = None

    header=None
    data=None

    fitsName=None
    Nx,Ny,Nz=None,None,None

    dimension=3#### by defaults
    saveTag=None

    secondXapplied=False
    secondYapplied=False
    #ylim= (-50,50)
    ylim= [-10,10]
    xlim=  [-200,300]


    figurePath="./cubeViewFigures/"

    drawCenter=True

    velValues=None
    prefix=""

    CO13lineCode="L"
    CO12lineCode="U"
    CO18lineCode="L2"


    def  __int__(self):
        pass

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

    def set_xlim(self,minV=-200,maxV=300):
        self.xlim=[minV,maxV ]
    def set_ylim(self,minK=-20,maxK=20):
        self.ylim=[minK,maxK ]


    def getSpectraByIndex(self,data, wcs, indexX, indexY):
        """
        paramters: data,dataHeader,indexX,indexY #### only for three dimensional
        This function is used to get a voxel value from a data cube

        v, km/s
        """
        ####




        ##it is possible that the yindex,and xindex can exceed the boundary of spectrum

        spectra = data[:, indexY, indexX]
        ##just for test
        # print  w.all_world2pix(0, 0, 0,0)
        # print data.shape[0]
        velocityIndex = range(data.shape[0])

        velocities = wcs.all_pix2world(indexX, indexY, velocityIndex, 0)[2] / 1000.


        ####


        #
        return spectra , velocities



    def getVTpoints(self,dataToSub = None ):
        """
        this function is used to convert the data cube into pints with coordinates of velocity and
        :return:
        """
        ylim = self.ylim
        if self.drawCenter:
            ### [15, 75]
            cropData = self.data[:,15:76,15:76]
            if dataToSub is not None:
                cropData = cropData - dataToSub[:, 15:76, 15:76]

        else:
            cropData= self.data
            if dataToSub is not None:
                cropData = cropData - dataToSub


        indexAll =np.where( np.isreal( cropData) )

        Nz,Ny,Nx=cropData.shape


        allValues = cropData[indexAll]




        #### find the linear map between z and velcoites
        zIndex = indexAll[0]
        spec0,vel0= self.getSpectraByIndex(self.data,self.wcs,0,0)
        velocites = vel0[0]+(vel0[1] - vel0[0]) *zIndex


        if ylim is not None:
            selectCriteria = np.logical_and(allValues>=min(ylim), allValues<=max(ylim))




        selectV = velocites[selectCriteria]
        selectTmb = allValues[selectCriteria]

        if 0:

            selectV=np.insert(selectV,0,velocites[0])
            selectTmb=np.insert(selectTmb,0, np.nan )

            selectV=np.insert(selectV,0,velocites[0])
            selectTmb=np.insert(selectTmb, 0, np.nan )


            selectV=np.insert(selectV,0,velocites[-1])
            selectTmb=np.insert(selectTmb,0, np.nan )







        dataPoints = np.asarray( [ selectV ,  selectTmb ])
        dataPointsT = np.transpose(dataPoints)

        #dataPoints=np.insert(dataPoints,0,[velocites[0],np.nan]) #### inorder to
        #dataPoints=np.insert(dataPoints,0,[velocites[-1],np.nan])

        return   velocites, dataPointsT, dataPoints
        #### convert indexZ to velocities

        #reject,reject,velocites= self.wcs.wcs_pix2world( indexAll[2], indexAll[1],indexAll[0],0)  # too slow

        #find the linear

        #return velocites,allValues


    def velEdges(self):
        """

        :param corordinates:
        :return:
        """

        _,_,velValues = self.wcs.wcs_pix2world(0,0,np.arange(self.Nz),0)

        velValues = velValues /1000.

        dv=  velValues[1] - velValues[0]


        righEdges= velValues+dv/2.

        allEdges = np.insert(righEdges,0, velValues[0]-dv/2. )
        if allEdges[0] >allEdges[1]:
            allEdges=  allEdges[::-1]
        return  allEdges

    def getStr(self,X):
        """

        :param X:
        :return:
        """
        return ["%.0f" % z for z in X]



    def tick_function(self,X):




        V = 1 / (1 + X)
        return ["%.3f" % z for z in V]

    def drawAllPoints(self ,ylim=None,redo= False ,savePath=None,dT=0.1 ):

        if ylim is not None:
            self.ylim= tuple(ylim)



        fitsBase =os.path.basename(self.fitsName)
        saveTag = os.path.splitext(fitsBase)[0]
        saveName = saveTag + "_allSpectra.png"

        if not redo and os.path.isfile(saveName):
            print(saveName," has been processed.")
            return


        velocities, VTdata ,   dataPoints = self.getVTpoints( )
        velEdges = self.velEdges( )


        sizeT =  abs(  self.ylim[0] - self.ylim[1] )/dT
        tmbEdges = np.linspace( self.ylim[0], self.ylim[1], int(sizeT) )



        heatData , xedges, yedges = np.histogram2d( dataPoints[0], dataPoints[1],bins=(velEdges,tmbEdges) )
        heatData[heatData==0] = np.nan

        heatData =  heatData.T
        X, Y = np.meshgrid(xedges, yedges)




        fig = plt.figure(figsize=(8, 4))
        plt.subplots_adjust(wspace=0.2)

        fontsize = 13

        rc('text', usetex=True)
        rc('font', **{'family': 'sans-serif', 'size': fontsize, 'serif': ['Helvetica']})
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{tgheros} \usepackage{sansmath} \sansmath'
        #plt.rcParams['axes.facecolor'] = 'white'
        ax = fig.add_subplot(1, 1, 1 )
        #ax2 = ax.twiny()

        ax2 = ax.secondary_xaxis('top')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

        #ax.set_xlabel(r'$ \mathit{V}_{\rm LSR} (\rm km \ s^{-1})$', fontsize=fontsize)
        #ax.set_ylabel(r'$\mathit{T}_{\rm mb} \ (\rm K)$', fontsize=fontsize)

        ax.set_xlabel(r'$\mathit V_{\rm LSR}$ (km s$^{-1}$)', fontsize=fontsize)
        ax.set_ylabel(r'$\mathit V_{\rm mb}$ (K)', fontsize=fontsize)

        #ax.set_xlim( )
        ax.set_ylim(self.ylim )
        #ax.set_xlim( [self.velValues[0] , self.velValues[-1] ] )
        ax.set_xlim( [self.xlim[0] , self.xlim[-1] ] )

        #ax.axvspan(-300, 300, alpha=.3, color='gray')

        ax.pcolormesh(X, Y, heatData , cmap="jet")

        at = AnchoredText(fitsBase, loc=1, frameon=False, prop={"color": "black", "size": fontsize - 3})
        ax.add_artist(at)
        if 1:
            x_indices = np.arange(len(self.velValues)) #
            x_centers = (xedges[:-1] + xedges[1:]) / 2  # Center of each cell
            step = max(1, len(self.velValues) // 10)  # Show about 10 ticks

            show_indices = x_indices[::step]
            show_positions = x_centers[::step]

            ax2.set_xticks(show_positions)
            ax2.set_xticklabels(show_indices)
            ax2.set_xlabel(r"Channel number")


        if 0:

            ax2.set_xlim(ax.get_xlim())

            ####

            ticks_positions = ax.get_xticks()
            print(ticks_positions,"What is the ticks location?")
            new_tick_locationsChannel= np.arange(0,18000,500)

            #ticks_positions
            _,_,new_tick_locationsVel =  self.wcs.wcs_pix2world(0,0,new_tick_locationsChannel,0)
            new_tick_locationsVel= new_tick_locationsVel/1000.



            selectCriteria = np.logical_and( new_tick_locationsVel>=min(self.xlim),  new_tick_locationsVel<=max(self.xlim) )

            new_tick_locationsChannel=new_tick_locationsChannel[ selectCriteria]

            new_tick_locationsVel=new_tick_locationsVel[ selectCriteria]


            ax2.set_xticks(new_tick_locationsVel)
            ax2.set_xticklabels(self.getStr(new_tick_locationsChannel))
            ax2.set_xlabel(r"Channel number")


        ax.axhline(y=0,lw=0.5,color="black",ls="--")


        saveTag = os.path.splitext(fitsBase)[0]
        saveName = saveTag+"_allSpectra.png"
        if savePath is  None :

            if not os.path.isdir(self.figurePath):
                os.mkdir(self.figurePath)

            saveName=os.path.join(self.figurePath,saveName )
        else:
            saveName=os.path.join( savePath ,saveName)




        plt.savefig(saveName,  bbox_inches='tight', dpi=600)
        plt.close()
        print("Saved as ", saveName)

        return


    def getSaveTag(self):
        if self.fitsName is None:
            return None


        saveBasename = os.path.basename(self.fitsName)
        self.saveTag= os.path.splitext(saveBasename)[0]


    def set_figurePath(self,figurePath):
        """

        """
        if os.path.isdir(figurePath):
            self.figurePath = figurePath
        else:
            print("The path does not exist,",figurePath)





    def set_FITS(self,fitsName):
        """

        :param fitsName:
        :return:
        """


        if fitsName is None:
            return

        else:
            self.fitsName=fitsName

            self.getSaveTag()
            data,header= self.readFITS(self.fitsName)

            self.data = data

            if len(data.shape)==4:
                self.data = data[0] # the fourth dimension is

            self.dimension= len( self.data.shape)

            if self.dimension == 3:
                self.Nz, self.Ny, self.Nx= self.data.shape


            if self.dimension == 2 :
                self.Ny, self.Nx= self.data.shape

            #####


            self.header = header
            self.wcs = WCS(self.header ,  naxis= self.dimension  )
            self.wcs.wcs.bounds_check(False, False)  # for molecular clouds toward the anti Galacic center

            self.wcs2D = WCS(self.header ,  naxis= 2  ) ####
            self.wcs2D.wcs.bounds_check(False, False)  # for molecular clouds toward the anti Galacic center

            _, _, self.velValues  = self.wcs.wcs_pix2world(0, 0, np.arange(self.Nz), 0)
            self.velValues = self.velValues  /1000.


            self.xlim=[np.min(self.velValues),np.max(self.velValues) ]




    def  runFITS(self,fitsName,redo=1):
        """

        :param fitsName:
        :return:
        """
        self.set_FITS( fitsName )
        self.drawAllPointsTest(redo=redo)

    def runPath(self,fitsToPath, lineCode="U", redo=1,prefix=""  ):
        """
        piple line for path
        :param fitsToPath:
        :return:
        """

        saveFolder=self.figurePath

        #saveFolder =  os.path.join( savePath, "viewFigures")

        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder)



        #############
        searchStr= os.path.join(fitsToPath, "{}*{}.fits".format(prefix, lineCode ))

        allFITS = glob.glob( searchStr )



        for eachFile in allFITS:
            print("Processing ",eachFile)
            self.set_FITS(eachFile)

            self.drawAllPointsTest(redo= redo ,savePath = saveFolder   )

            gc.collect()




    def zzz(self):
        pass


__author__ = 'Qing-Zeng, 2022 Apr 11'
__info__= 'Modified 2022, Oct 25, to make is a single file, indepdent of myFITS, and myDrawFITS'

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')
    parser.add_argument('-f', '--filename', help='Input source name', required=False)
    parser.add_argument('-p', '--pathname', help='Input path name', required=False,    default='.')
    parser.add_argument('-outpath', '--outpath', help='Outpath name for png files', required=False,    default='.')

    parser.add_argument('-center', '--center', help='Only draw Center region, cut edge by 15 pixels ', required=False,type=int, default= 1 ) ### by defaults

    parser.add_argument('-line', '--line', help='line code (12CO: U, 13CU: L,C18O:L2)', required=False,default="U")
    parser.add_argument('-redo', '--redo', help='Whether redo the process', required=False,type=int, default= 1 )

    parser.add_argument('-minT', '--minT', help='minimum of Y axis default(-30) ', required=False, type=float, default= -30 ) ### by defaults
    parser.add_argument('-maxT', '--maxT', help='maximum of Y axis default(30)', required=False, type=float, default= 30 ) ### by defaults


    args = parser.parse_args()

    doDraw = drawCube()
    doDraw.drawCenter = args.center
    doDraw.ylim= (args.minT,args.maxT)
    doDraw.figurePath=args.outpath

    if args.filename is None: # single file is previlage

        if os.path.isdir(args.pathname):
            print("Processing path {}".format(args.pathname))

            dataPath =  args.pathname
            searchStr= os.path.join(dataPath, "*{}.fits".format(args.line))

            allFITS = glob.glob( searchStr )

            if len(allFITS) ==0:
                print("No files found for line {} (use -line to change) (i.e., *{}.fits)".format(args.line,args.line))


            for eachFile in allFITS:
                print("Processing ",eachFile)
                doDraw.set_FITS(eachFile)

                doDraw.drawAllPointsTest(redo=args.redo )

                gc.collect()




        else:
            print("Path does not exist, ", args.pathname )
            sys.exit()


    else:

        if os.path.isfile(args.filename):
            print("Drawing all spectra of file {}".format(args.filename))
            doDraw.set_FITS(args.filename)
            doDraw.drawAllPointsTest(redo=args.redo)

        else:
            print("File {} does not exist!!!".format(args.filename))
            sys.exit()

