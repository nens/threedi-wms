# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

# Register your backends here
BLUEPRINTS = [
    'threedi_wms.threedi',
    'threedi_wms.rasterinfo',
]

# Directories
BUILDOUT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
)
CACHE_DIR = os.path.join(BUILDOUT_DIR, 'var', 'cache')
CELERY_DIR = os.path.join(BUILDOUT_DIR, 'var', 'celery')
DATA_DIR = os.path.join(BUILDOUT_DIR, 'var', 'data')
LOG_DIR = os.path.join(BUILDOUT_DIR, 'var', 'log')

# Celery
CELERY_DB = os.path.join(CELERY_DIR, 'celerydb.sqlite')

# default settings, overriden on server by local settings
USE_CACHE = False  # redis
CACHE_PREFIX = 'subgrid:10000'
THREEDI_SUBGRID_ID = 'subgrid:10000'
THREEDI_STANDALONE_SUBGRID_MACHINE = False

# SENTRY_DSN = None  # TODO: fill

# redis settings for reporting threedi-wms status messages like busy, not busy,
# and current timestep; override in generated local settings if needed
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_STATE_DB = 0
REDIS_NODE_MAPPING_DB = 2

WMS_BUSY_THRESHOLD = 2  # 2 seconds

# import local settings
try:
    from server.localsettings import *
except ImportError:
    pass
