# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import logging

import celery

from osgeo import gdal

from gislib import rasters
from gislib import projections

from server import config

from threedi_wms.threedi import quads
from threedi_wms.threedi import utils

# autocreate celery db dir
try:
    os.makedirs(os.path.dirname(config.CELERY_DB))
except OSError:
    pass  # No problem.

app = celery.Celery()
app.conf.update(
    BROKER_URL='sqla+sqlite:///{}'.format(config.CELERY_DB),
)

# set the SENTRY_DSN as environment variable; this is needed to use the
# SentryHandler logging handler by celery tasks
os.environ['SENTRY_DSN'] = config.SENTRY_DSN
logger = logging.getLogger('')


@app.task
def make_pyramid(layer):
    """ Build a pyramid from a dataset. """
    logger.info('Building pyramid for {}'.format(layer))
    # Get paths
    pyramid_path = utils.get_pyramid_path(layer)
    dataset_path = utils.get_bathymetry_path(layer)
    # Create pyramid
    try:
        pyramid = rasters.Pyramid(path=pyramid_path, compression='DEFLATE')
        if pyramid.has_data():
            logger.info('Pyramid has data for {}'.format(layer))
            return
        dataset = gdal.Open(str(dataset_path))

        logger.info("Pyramid path: %r" % pyramid_path)
        logger.info("Dataset path: %r" % dataset_path)
        # it defaults to rijksdriehoek (28992)
        bathy_srs = utils.get_bathymetry_srs(dataset_path)
        logger.info("Bathy srs: %r" % bathy_srs)
        dataset.SetProjection(projections.get_wkt(bathy_srs))
        pyramid.add(dataset)
    except rasters.LockError:
        logger.info('Pyramid busy for {}'.format(layer))
        return
    logger.info('Pyramid completed for {}'.format(layer))


@app.task
def make_monolith(layer):
    """ Build a monolith from a netcdf. """
    logger.info('Building monolith for {}'.format(layer))
    # Get paths
    monolith_path = utils.get_monolith_path(layer)
    netcdf_path = utils.get_netcdf_path(layer)
    bathy_path = utils.get_bathymetry_path(layer)
    projection = utils.get_bathymetry_srs(bathy_path)
    # if projection is not None:
    #     projection = int(projection)
    logger.info('Monolith projection is {}'.format(projection))
    # Create monolith
    try:
        monolith = rasters.Monolith(path=monolith_path, compression='DEFLATE')
        if monolith.has_data():
            logger.info('Monolith has data for {}'.format(layer))
            return
        monolith.add(
            quads.get_dataset(path=netcdf_path, projection=projection))
    except rasters.LockError:
        logger.info('Monolith busy for {}'.format(layer))
        return
    logger.info('Monolith completed for {}'.format(layer))


@app.task
def test_logging():
    """
    Use this task from the command-line to check whether log messages arrive
    into logstash and/or sentry depending on your logging configuration, of
    course.

    Usage:
    $ cd src/threedi-wms
    $ bin/python
    $ from threedi_wms.threedi.tasks import test_logging
    >>> test_logging.delay()

    Then look in logstash and/or sentry whether the log messages are there.

    """
    loglevels = ['debug', 'info', 'error']
    for loglevel in loglevels:
        getattr(logger, loglevel)(
            "test %s message from celery" % loglevel)
    try:
        1 / 0
    except:
        logger.exception("1 / 0 from test_logging celery task")
    a = 4 / 2
    sentry_dsn = os.environ.get('SENTRY_DSN', None)
    if sentry_dsn:
        logger.debug("env var SENTRY_DSN = %s" % sentry_dsn)
    else:
        logger.error("env var SENTRY_DSN: NOT SET! should not happen!")
    print("4 / 2 = %s" % a)  # test a non-logger message
