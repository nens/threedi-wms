
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from server.config import *
import os

BLUEPRINT_NAME = '3di'

DATA_DIR = os.path.join(DATA_DIR, BLUEPRINT_NAME)
CACHE_DIR = os.path.join(CACHE_DIR, BLUEPRINT_NAME)
