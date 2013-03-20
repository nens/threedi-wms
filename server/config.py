# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

# Directories
BUILDOUT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
)
DATA_DIR = os.path.join(BUILDOUT_DIR, 'var', 'data')
CACHE_DIR = os.path.join(BUILDOUT_DIR, 'var', 'cache')
CELERY_DIR = os.path.join(BUILDOUT_DIR, 'var', 'celery')

# Mapping packages to url parts
BLUEPRINTS = {
    'threedi',
}

# Celery
CELERY_DB = os.path.join(CELERY_DIR, 'celerydb.sqlite')

# Import local settings
try:
    from threedi_server.localconfig import *
except ImportError:
    pass
