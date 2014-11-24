
import redis

import config


class Status(object):
    """Set threedi-wms status messages like busy, not busy and timestep
    in redis. These messages can be used to be reported back to the end-user.

    """
    STATES = {
        0: 'not busy',
        1: 'busy'
    }

    def __init__(self):
        """Initialise redis connection."""
        self.rc = redis.Redis(
            host=config.REDIS_STATUS_HOST, port=config.REDIS_STATUS_PORT,
            db=config.REDIS_STATUS_DB)

    def update_timestep(self, timestep):
        """Write timestep to redis."""
        # TODO: create config.THREEDI_SUBGRID_ID; is same as
        # config.CACHE_PREFIX
        self.rc.set("%s:wms:timestep" % config.THREEDI_SUBGRID_ID, timestep)

    def update_state(self, state):
        """Write state to redis."""
        self.rc.set("%s:wms:state" % config.THREEDI_SUBGRID_ID, state)
