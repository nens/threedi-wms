# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import fcntl
import socket
import struct

import redis

import flask

from server import config


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
        # get the subgrid id by the machine's ip address
        rc = redis.Redis(
            host=config.REDIS_HOST, port=config.REDIS_PORT,
            db=config.REDIS_STATE_DB)
        ip_address = get_ip_address()
        subgrid_id = rc.get('subgrid:%s:id' % ip_address)
        return subgrid_id
    else:
        # not standalone; return the default subgrid id
        return config.THREEDI_SUBGRID_ID
