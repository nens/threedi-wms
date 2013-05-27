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
    #wktline = "LINESTRING (624509.038879959494807 6804205.040622464381158,\
    #624606.396697581047192 6803551.216062948107719)"
    #wktline = "LINESTRING (624185.945052515366115 6803547.505825876258314, \
    #625049.321208689478226 6803997.485771679319441, \
    #625122.488679551752284 6803865.78432412724942)"
    #wktline = "LINESTRING (624241.882649042527191 6803781.921267291530967, \
    #624459.548982131062075 6803781.921267291530967)"
    #src_srs = 900913
    if request.method == 'GET':
        src_srs = int(request.values['srs'])
        wktline = request.values['geom']
    profile = [rasterinfo.get_profile(wktline, src_srs)]
    return jsonify(profile=profile)
