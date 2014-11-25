# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import redis

import config


class Status(object):
    """
    Set threedi-wms status messages like busy, not busy and current timestep
    in redis state database. These messages can then be forwarded to the
    end-user.
    """
    STATE_NOT_BUSY = 0
    STATE_BUSY = 1

    def __init__(self):
        """Initialise a redis client connection to the state database."""
        self.rc = redis.Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT,
            db=config.REDIS_STATE_DB)

    def update_timestep(self, timestep):
        """Write timestep to redis."""
        self.rc.set("%s:wms_timestep" % config.THREEDI_SUBGRID_ID, timestep)

    def update_state(self, state):
        """Write state to redis."""
        self.rc.set("%s:wms_state" % config.THREEDI_SUBGRID_ID, state)
