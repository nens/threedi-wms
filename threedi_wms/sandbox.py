# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging

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
    parser.add_argument('targetpath', metavar='TARGET')
    parser.add_argument('-t', '--timestep', metavar='TIMESTEP', type=int)
    # Add arguments here.
    return parser


def command(sourcepath, targetpath, timestep=None):
    """ Do something spectacular. """
    quaddataset = quads.get_dataset(sourcepath)
    quaddata = quaddataset.ReadAsArray()

    if timestep is None:
        logging.debug('Calculating maximum flow velocity.')
        with Dataset(sourcepath) as dataset:
            fvx = dataset.variables['ucx'][:].max(0)
            fvy = dataset.variables['ucy'][:].max(0)
    else:
        with Dataset(sourcepath) as dataset:
            logging.debug('Calculating flow velocity for time = {}.'.format(
                dataset.variables['time'][timestep],
            ))
            fvx = dataset.variables['ucx'][:][timestep]
            fvy = dataset.variables['ucy'][:][timestep]

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
    fvdataset.SetGeoTransform(quaddataset.GetGeoTransform())
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
