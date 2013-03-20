# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import sys

import celery

from server import config

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
)

@celery.task
def build_pyramid():
    """ Build a pyramid. """
    print("I'm going to build the pyramids.")
