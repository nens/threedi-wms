# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from flask import jsonify, request, abort

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

    Example: (setup data for Utrecht center)
    run Flask dev server on localhost and go to
    `http://127.0.0.1:5000/rasterinfo/profile?geom=LINESTRING(566582.6327506%206816233.9453951,%20570156.06382243%206816233.9453951)&epsg=900913`
    """
    if not 'epsg' in request.values or 'geom' not in request.values:
        abort(400)
    src_epsg = int(request.values['epsg'])
    wktline = request.values['geom']
    profile = [rasterinfo.get_profile(wktline, src_epsg)]
    return jsonify(profile=profile)
