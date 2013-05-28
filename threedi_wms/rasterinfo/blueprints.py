# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from flask import jsonify, request

from server import blueprints

from threedi_wms.rasterinfo import config
from threedi_wms.rasterinfo import rasterinfo


rasterapp = blueprints.Blueprint(name=config.BLUEPRINT_NAME,
                                 import_name=__name__)


@rasterapp.route('/')
def index():
    return '<h1>Rasterinfo</h1><a href="profile">Profile tool</a>'


@rasterapp.route('/profile', methods=['GET'])
def rasterprofile():
    """
    Return json with [distance, raster values] according to request.
    """
    if request.method == 'GET':
        src_epsg = int(request.values['epsg'])
        wktline = request.values['geom']
    profile = [rasterinfo.get_profile(wktline, src_epsg)]
    return jsonify(profile=profile)
