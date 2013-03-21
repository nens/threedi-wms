# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging
import os

import celery

from server import config
from server import loghelper

# Autocreate celery db dir
try:
    os.makedirs(os.path.dirname(config.CELERY_DB))
except OSError:
    pass  # No problem.

# Configure celery
app = celery.Celery()
app.conf.update(
    BROKER_URL='sqla+sqlite:///{}'.format(config.CELERY_DB),
)

# Setup logging
loghelper.setup_logging(logfile_name='tasks.log')


# Temporary test task here.
@app.task
def build_pyramid():
    """ Build a pyramid. """
    logging.info("I'm going to build the pyramids.")
