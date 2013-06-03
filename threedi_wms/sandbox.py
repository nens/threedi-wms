# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse

from netCDF4 import Dataset
from osgeo import gdal

import numpy as np

from threedi_wms.threedi import quads


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('sourcepath', metavar='SOURCE')
    parser.add_argument('timestep', metavar='TIMESTEP', type=int)
    parser.add_argument('targetpath', metavar='TARGET')
    # Add arguments here.
    return parser


def command(sourcepath, targetpath, timestep):
    """ Do something spectacular. """
    quaddataset = quads.get_dataset(sourcepath)
    quaddata = quaddataset.ReadAsArray()

    with Dataset(sourcepath) as dataset:
        fvx = dataset.variables['ucx'][:][timestep]
        fvy = dataset.variables['ucy'][:][timestep]
        #import ipdb; ipdb.set_trace() 
    
    # Create linear array
    fv = np.ma.array(
        np.empty(fvx.size + 1),
        mask=True,
    )

    # Fill with flowvelocity from netcdf
    fv[0:-1] = np.sqrt(fvx ** 2 + fvy ** 2)

    # Create mem dataset
    mem_driver = gdal.GetDriverByName(b'mem')
    fvdataset = mem_driver.Create(
        b'',
        quaddataset.RasterXSize,
        quaddataset.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    fvband = fvdataset.GetRasterBand(1)
    fvband.Fill(-999)
    fvband.SetNoDataValue(-999)
    fvband.WriteArray(fv[quaddata].filled(-999))
    
    # Create asciifile
    asc_driver = gdal.GetDriverByName(b'aaigrid')
    asc_driver.CreateCopy(
        targetpath,
        fvdataset,
        options=[b'DECIMAL_PRECISION=3']
    )

def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
