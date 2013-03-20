
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import server

import os

BLUEPRINT_NAME = '3di'

DATA_DIR = os.path.join(server.config.DATA_DIR, BLUEPRINT_NAME)
CACHE_DIR = os.path.join(server.config.CACHE_DIR, BLUEPRINT_NAME)

# Mapping packages to url parts
BLUEPRINTS = {
    'blueprint_3di',
}

GEOSERVER_LAYER = 'nl:ahn_rgba'

class FlaskConfig(object):
    """ Flask configuration container. """
    DEBUG = True

# Import local settings
try:
    from threedi_server.localconfig import *
except ImportError:
    pass
