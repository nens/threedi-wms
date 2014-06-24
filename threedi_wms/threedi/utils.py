# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import logging

try:
    from osgeo import gdal
except ImportError:
    import gdal

try:
    from osgeo import osr
except ImportError:
    import osr

import numpy as np
import redis

from gislib import projections

from threedi_wms.threedi import config

logger = logging.getLogger(__name__)

rc = redis.Redis()


def get_netcdf_path(layer):
    """ Return path to netcdf from layer. """
    name = os.path.join(config.DATA_DIR, layer, 'subgrid_map.nc')
    if not os.path.exists(name):
        print('expected file not found: %s' % name)
    return name
    # # Look for netcdf
    # for name in names:
    #     if os.path.splitext(name)[1].lower() == '.nc':
    #         return os.path.join(config.DATA_DIR, layer, name)


def get_netcdf_path_flood(layer):
    """ Return path to floodfill netcdf from layer. """
    name = os.path.join(config.DATA_DIR, layer, 'floodfill.nc')
    print(name)
    if not os.path.exists(name):
        print('expected floodfill file not found: %s' % name)
    return name


def get_bathymetry_path(layer):
    """
    Return path to bathymetry from layer.

    Prefers geotiff above aaigrid.
    """
    names = os.listdir(os.path.join(config.DATA_DIR, layer))
    # Look for geotiff
    for name in names:
        if os.path.splitext(name)[1].lower() in ('.tif', '.tiff'):
            return os.path.join(config.DATA_DIR, layer, name)
    for name in names:
    # Look for aaigrid
        if os.path.splitext(name)[1].lower() in ('.asc'):
            return os.path.join(config.DATA_DIR, layer, name)


def get_pyramid_path(layer):
    """ Return pyramid path. """
    return os.path.join(config.CACHE_DIR, layer, 'pyramid')


def get_monolith_path(layer):
    """ Return monolith path. """
    return os.path.join(config.CACHE_DIR, layer, 'monolith')


def get_bathymetry_srs(filename):
    """Return srs from bathymetry, None if not set"""
    ds = gdal.Open(filename)
    return projections.get_wkt(ds.GetProjection())
    # src = osr.SpatialReference()
    # src.ImportFromWkt(ds.GetProjection())
    # result = src.GetAttrValue(str('PROJCS|AUTHORITY'), 1)  # None or '22234'
    # ds = None  # Close dataset
    # return result


def get_loaded_model():
    """Return the loaded_model (slug) from redis)."""
    threedi_subgrid_id = config.CACHE_PREFIX
    return rc.get('%s:loaded_model' % threedi_subgrid_id)


def classify(data, classes):
    """Classify data (arrays or lists) based on its values.

    :param data - array of floats/integers (1-dimensional) or list of
        floats/integers
    :param classes - dictionary of classifier keys with their value range, e.g.
    classes = {
        -7: (-3.00, -3.00),  # will be used for lower as well
        -6: (-3.00, -1.00),
        -5: (-1.00, -0.50),
        -4: (-0.50, -0.30),
        -3: (-0.30, -0.10),
        -2: (-0.10, -0.05),
        -1: (-0.05, -0.01),
         0: (-0.01, 0.01),
         1: (0.01, 0.05),
         2: (0.05, 0.10),
         3: (0.10, 0.30),
         4: (0.30, 0.50),
         5: (0.50, 1.00),
         6: (1.00, 3.00),
         7: (3.00, 3.00),  # will be used for higher as well
    }

    :return array of classifiers

    """
    class_dtype = np.dtype({'names': ['classes', 'limits'],
                            'formats': ['i8', '2f8']})
    class_array = np.array(classes.items(), dtype=class_dtype)
    class_array.sort()

    fp = (class_array['classes'] * np.ones((2, 1))).transpose().ravel()
    xp = class_array['limits'].ravel()

    return np.int8(np.interp(x=data, xp=xp, fp=fp))
