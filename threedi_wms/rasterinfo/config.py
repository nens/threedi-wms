
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

from server import config


BLUEPRINT_NAME = 'rasterinfo'

DATA_DIR = os.path.join(config.DATA_DIR, BLUEPRINT_NAME)
CACHE_DIR = os.path.join(config.CACHE_DIR, BLUEPRINT_NAME)
PYRAMID_PATH = os.path.join(DATA_DIR, 'pyramid')
