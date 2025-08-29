from mwispDBSCANV5dev import MWISPDBSCAN

#before running the script, you may need to install some packages
"""
including but not limited to:

conda install astropy
pip install radio_beam spectral_cube 
pip install  pvextractor progressbar termcolor pyregion corner mip pwlf colorama
"""
if 1:
    doDBSCAN = MWISPDBSCAN()
    doDBSCAN.setConfigFile("dbscanRecords.ini") #a file to save the pat of files used, and the path save DBSCAN result
    # this is useful when you need to recall the DBSCAN results.

    doDBSCAN.extraTag="catExample" # a tag for multiple cases of DBSCAN running
    doDBSCAN.rawCOFITS="testQ3.fits"# the CO data cube
    doDBSCAN.rmsFITS="testQ3_rms.fits" # 2D rms data

    # Some times, the data cube is too large the cloud parameter takes too long, the system crashes, you the DBSCAN part the cloud parameter
    #part can be run step by step. redoDBSCAN= True means the previouse results will be overwrite
    #onlyCat=Ture mean only do cat calculation, using the previous DBSCAN results
    # writeconfig=True, means save the DBSCAN to the configureFile (dbscanRecords.ini)
    doDBSCAN.pipeLine(writeconfig=True, onlyCat=False, redoDBSCAN= True )


if 0:#example to recall previeouly run DBSCAN
    doDBSCAN = MWISPDBSCAN()
    #doDBSCAN.setConfigFile("dbscanRecords.ini")  # a file to save the pat of files used, and the path save DBSCAN result
    # this is useful when you need to recall the DBSCAN results.

    doDBSCAN.getDBSCANcase("testQ3dbscanS2P4Con1catExample","dbscanRecords.ini" )