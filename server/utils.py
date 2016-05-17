# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging

import numpy as np
import flask


logger = logging.getLogger('')


def get_parameters():
    """ Return the request parameters with lowercase keys. """
    return {k.lower(): v for k, v in flask.request.args.items()}


def fetch_subgrid_id():
    """Return the subgrid id."""
    try:
        subgrid_id = os.environ['SUBGRID_ID']
    except KeyError:
        logger.exception("Missing SUBGRID_ID env var.")
        raise
    else:
        logger.info(
            "Got subgrid_id {} from env var `SUBGRID_ID`.".format(
                subgrid_id))
        return subgrid_id


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
