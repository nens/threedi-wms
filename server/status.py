# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import logging
import time

import redis

from server import config
from server import utils


logger = logging.getLogger('')


def to_timestamp(dt, epoch=datetime(1970, 1, 1)):
    """Seconds since epoch for the given datetime."""
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 1e6


def current_timestamp():
    """Seconds from epoch for the current UTC time."""
    now = datetime.utcnow()
    return to_timestamp(now)


class StateReporter(object):
    """
    Set threedi-wms status messages in redis state database. These messages
    can then be forwarded to the end-user.

    """
    def __init__(self):
        """Initialise a redis client connection to the state database."""
        self.rc = redis.Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT,
            db=config.REDIS_STATE_DB)
        # get the correct subgrid id
        subgrid_id = utils.fetch_subgrid_id()
        while not subgrid_id:
            logger.info(
                '[StateReporter] waiting for a subgrid id from redis...')
            time.sleep(1)
            subgrid_id = utils.fetch_subgrid_id()
        self.redis_key = subgrid_id

    def set_timestep(self, timestep):
        """Write timestep to redis."""
        self.rc.set('%s:wms_timestep' % self.redis_key, timestep)

    def set_busy(self):
        """
        Increase wms_busy_workers.

        INCR is used, because we can have multiple wms workers, so we know
        when wms_busy_workers is > 0 at least one of the workers is busy.

        Also we need to know whether the previous number of busy wms workers
        was 0. In that case, we need to set the wms_busy_since timestamp. This
        timestamp is used as an interval after which wms is regarded as really
        busy, for example after a predefined number of seconds of
        wms_busy_workers being > 0. After that predefined number of  seconds a
        busy flag is set and forwarded to the client for further use in the
        client interface. The pipeline makes the get and incr commands atomic
        together.

        """
        pipe = self.rc.pipeline()
        pipe.get('%s:wms_busy_workers' % self.redis_key)
        pipe.incr('%s:wms_busy_workers' % self.redis_key)
        previous_busy_workers, _ = pipe.execute()
        if (previous_busy_workers is not None and
                int(previous_busy_workers) == 0):  # current == 1
            # switch from 0 to 1 busy wms worker: set wms_busy_since timestamp
            # to be used by handle_busy_flag
            self.rc.set('%s:wms_busy_since' % self.redis_key,
                        current_timestamp())

    def set_not_busy(self):
        """Decrease wms_busy_workers."""
        pipe = self.rc.pipeline()
        pipe.get('%s:wms_busy_workers' % self.redis_key)
        pipe.decr('%s:wms_busy_workers' % self.redis_key)
        previous_busy_workers, _ = pipe.execute()
        if (previous_busy_workers is not None and
                int(previous_busy_workers) == 1):  # current == 0
            # switch from 1 to 0 busy wms workers: remove wms_busy_since
            self.rc.delete('%s:wms_busy_since' % self.redis_key)
        # We suspect that sometimes the number of busy workers falls below 0.
        # We don't know the exact cause, but we need to adjust that, because
        # number of busy workers can never be below 0.
        if (previous_busy_workers is not None and
                int(previous_busy_workers) == 0):  # current == -1
            self.correct_negative_busy_workers()

    def get_busy_workers(self):
        """Return number of busy workers. Primarily for debug purposes."""
        return self.rc.get('%s:wms_busy_workers' % self.redis_key)

    @property
    def busy_duration(self):
        """Return number of seconds wms is considered busy or time in seconds
        the number of busy wms workers was not 0. Used for debug logging."""
        busy_since = self.rc.get('%s:wms_busy_since' % self.redis_key)
        if busy_since:
            current_ts = current_timestamp()
            return current_ts - float(busy_since)

    def reset_all(self):
        """
        Reset all state variables.

        This can be used upon start of the wms server. See app.py.
        """
        self.rc.set('%s:wms_busy_workers' % self.redis_key, 0)
        self.rc.set('%s:wms_timestep' % self.redis_key, None)

    def handle_busy_flag(self):
        """Check whether wms is busy longer than the WMS_BUSY_THRESHOLD. If so,
        set the wms_busy flag."""
        busy_since = self.rc.get('%s:wms_busy_since' % self.redis_key)
        if busy_since:
            current_ts = current_timestamp()
            if (current_ts - float(busy_since)) > config.WMS_BUSY_THRESHOLD:
                # now wms should be considered busy for the client as well
                # set the wms_busy flag to be used by the client
                self.rc.set('%s:wms_busy' % self.redis_key, 1)
            else:
                self.rc.delete('%s:wms_busy' % self.redis_key)

    def correct_negative_busy_workers(self):
        """
        We suspect that sometimes the number of busy workers falls below 0.
        We don't know the exact cause, but we need to adjust that, because
        number of busy workers can never be below 0.
        """
        self.rc.set('%s:wms_busy_workers' % self.redis_key, 0)
