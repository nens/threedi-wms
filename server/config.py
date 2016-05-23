# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import ast
import logging.config


class ImproperlyConfigured(Exception):
    pass


def env(key, default=None, required=False):
    """
    Retrieves environment variables and returns Python natives. The (optional)
    default will be returned if the environment variable does not exist.
    """
    try:
        value = os.environ[key]
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    except KeyError:
        if default or not required:
            return default
        raise ImproperlyConfigured(
            "Missing required environment variable '%s'" % key)


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
USE_CACHE = True  # redis
CACHE_PREFIX = 'subgrid:10000'
THREEDI_SUBGRID_ID = 'subgrid:10000'
THREEDI_STANDALONE_SUBGRID_MACHINE = True

SENTRY_DSN = ''

# redis settings for reporting threedi-wms status messages like busy, not busy,
# and current timestep
REDIS_HOST = env('REDIS_HOST_STATE', default='localhost', required=True)
REDIS_PORT = 6379
REDIS_STATE_DB = 0
REDIS_NODE_MAPPING_DB = 2

WMS_BUSY_THRESHOLD = 2  # 2 seconds

# add default logging to stdout
LOGGING = {
    'disable_existing_loggers': False,
    'version': 1,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    }
}
logging.config.dictConfig(LOGGING)

# import local settings
try:
    from server.localsettings import *  # noqa
    from server.localloggingsettings import *  # noqa
except ImportError:
    pass
