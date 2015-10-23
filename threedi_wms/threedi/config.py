# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

from server.config import *


BLUEPRINT_NAME = '3di'

DATA_DIR = os.path.join(DATA_DIR, BLUEPRINT_NAME)
CACHE_DIR = os.path.join(CACHE_DIR, BLUEPRINT_NAME)

STATIC_DIR = os.path.join(
    BUILDOUT_DIR, 'threedi_wms', 'threedi', 'static', '3di')
