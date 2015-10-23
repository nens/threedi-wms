# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import sys

import celery

from server import blueprints
from server import config

"""
This module is only used by the celery worker, and not by the
server. Every blueprint can have it's own celery app, which must connect
to the same broker as this worker module for it's tasks to be executed
by the servers worker.
"""

# vvv Fix for celery forking problem
os.environ['PYTHONPATH'] = ':'.join(sys.path)

# Autocreate celery db dir
try:
    os.makedirs(os.path.dirname(config.CELERY_DB))
except OSError:
    pass  # No problem.

# Configure celery
app = celery.Celery()
app.conf.update(
    BROKER_URL='sqla+sqlite:///{}'.format(config.CELERY_DB),
    # CELERYD_HIJACK_ROOT_LOGGER=False,
)

# Import the blueprints, any tasks in them get registered with celery.
blueprints.get_blueprints()
