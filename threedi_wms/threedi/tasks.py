# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging
import os

import celery

from osgeo import gdal

from gislib import raster

from server import config
from server import loghelper

from threedi_wms.threedi import quads
from threedi_wms.threedi import utils

# Autocreate celery db dir
try:
    os.makedirs(os.path.dirname(config.CELERY_DB))
except OSError:
    pass  # No problem.

# Configure celery
app = celery.Celery()
app.conf.update(
    BROKER_URL='sqla+sqlite:///{}'.format(config.CELERY_DB),
)

# Setup logging
loghelper.setup_logging(logfile_name='tasks.log')


@app.task
def make_pyramid(layer):
    """ Build a pyramid from a dataset. """
    logging.info('Building pyramid for {}'.format(layer))
    # Get paths
    pyramid_path = utils.get_pyramid_path(layer)
    dataset_path = utils.get_bathymetry_path(layer)
    # Create pyramid
    try:
        pyramid = raster.Pyramid(path=pyramid_path, compression='DEFLATE')
        if pyramid.has_data():
            logging.info('Pyramid has data for {}'.format(layer))
            return
        pyramid.add(gdal.Open(str(dataset_path)))
    except raster.LockError:
        logging.info('Pyramid busy for {}'.format(layer))
        return
    logging.info('Pyramid completed for {}'.format(layer))


@app.task
def make_monolith(layer):
    """ Build a monolith from a netcdf. """
    logging.info('Building monolith for {}'.format(layer))
    # Get paths
    monolith_path = utils.get_monolith_path(layer)
    netcdf_path = utils.get_netcdf_path(layer)
    # Create monolith
    try:
        monolith = raster.Monolith(path=monolith_path, compression='DEFLATE')
        if monolith.has_data():
            logging.info('Monolith has data for {}'.format(layer))
            return
        monolith.add(quads.get_dataset(path=netcdf_path))
    except raster.LockError:
        logging.info('Monolith busy for {}'.format(layer))
        return
    logging.info('Monolith completed for {}'.format(layer))
