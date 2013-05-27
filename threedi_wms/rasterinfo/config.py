
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from server import config
import os

BLUEPRINT_NAME = 'rasterinfo'

DATA_DIR = os.path.join(config.DATA_DIR, BLUEPRINT_NAME)
CACHE_DIR = os.path.join(config.CACHE_DIR, BLUEPRINT_NAME)
