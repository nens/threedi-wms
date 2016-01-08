# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import fcntl
import logging
import socket
import struct

import numpy as np
import flask
import redis

from server import config

logger = logging.getLogger('')


def get_parameters():
    """ Return the request parameters with lowercase keys. """
    return {k.lower(): v for k, v in flask.request.args.items()}


def get_ip_address(ifname='eth0'):
    """Return the current machine's ip address depending on the given
    interface name."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


def fetch_subgrid_id():
    """Return the subgrid id.

    If standalone rekenmachine, use the machine's ip address to get the subgrid
    id.

    """
    if config.THREEDI_STANDALONE_SUBGRID_MACHINE:
        # standalone
        # some machine manager backends set the subgrid id as an environment
        # variable
        if os.environ.get('SUBGRID_ID'):
            subgrid_id = os.environ['SUBGRID_ID']
            logger.info(
                "Got subgrid_id {} from env var `SUBGRID_ID`.".format(
                    subgrid_id))
        else:
            # get the subgrid id by the machine's ip address
            rc = redis.Redis(
                host=config.REDIS_HOST, port=config.REDIS_PORT,
                db=config.REDIS_STATE_DB)
            ip_address = get_ip_address()
            subgrid_id = rc.hget('subgrid_ip_to_id', ip_address)
            logger.info(
                "Got subgrid_id {} by ip {} from redis.".format(
                    subgrid_id, ip_address))
        return subgrid_id
    else:
        # not standalone; return the default subgrid id
        return config.THREEDI_SUBGRID_ID


def logical_or_reduce(arrays):
    """Apply np.logical_or to one or more arrays.

    This is similar to np.logical_or.reduce, but much faster for some reason
    (possibly because logical_or.reduce initializes arrays every time).

    Args:
        arrays: a list of arrays
    """
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.logical_or(result, arr)
    return result
