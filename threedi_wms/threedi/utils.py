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

import redis

from gislib import projections

from server import config as server_config
from server import utils as server_utils

from threedi_wms.threedi import config

logger = logging.getLogger('')

rc = redis.Redis(
    host=server_config.REDIS_HOST_STATE, port=server_config.REDIS_PORT,
    db=server_config.REDIS_DB_STATE)


def get_netcdf_path(layer):
    """ Return path to netcdf from layer. """
    name = os.path.join(config.DATA_DIR, layer, 'subgrid_map.nc')
    if not os.path.exists(name):
        print('expected file not found: %s' % name)
    return name


def get_netcdf_path_flood(layer):
    """Return path to floodfill netcdf from layer."""
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
        # look for aaigrid
        if os.path.splitext(name)[1].lower() in ('.asc'):
            return os.path.join(config.DATA_DIR, layer, name)


def get_pyramid_path(layer):
    """Return pyramid path."""
    return os.path.join(config.CACHE_DIR, layer, 'pyramid')


def get_monolith_path(layer):
    """Return monolith path."""
    return os.path.join(config.CACHE_DIR, layer, 'monolith')


def get_bathymetry_srs(filename):
    """Return srs from bathymetry, None if not set."""
    ds = gdal.Open(filename)
    return projections.get_wkt(ds.GetProjection())


def get_loaded_model():
    """Return the loaded_model (slug) from redis)."""
    subgrid_id = server_utils.fetch_subgrid_id()
    return rc.get('%s:loaded_model' % subgrid_id)
