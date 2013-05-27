# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from flask import jsonify

from server import blueprints

from threedi_wms.rasterinfo import config


rasterapp = blueprints.Blueprint(name=config.BLUEPRINT_NAME,
                                 import_name=__name__)


@rasterapp.route('/')
def index():
    return "<h1>hello</h1>"


@rasterapp.route('/profile', methods=['GET'])
def rasterprofile():
    """
    Return json with [distance, raster values] according to request.
    """
    elevationprofile = {'0': [1, 3]}
    return jsonify(elevationprofile)
