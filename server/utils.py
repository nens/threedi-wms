# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import flask

def get_parameters():
    """ Return the request parameters with lowercase keys. """
    return  {k.lower(): v for k, v in flask.request.args.items()}
