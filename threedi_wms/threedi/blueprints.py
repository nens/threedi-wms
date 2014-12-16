# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

import flask
import numpy as np
import sys

from server import blueprints
from server import utils

from threedi_wms.threedi import responses
from threedi_wms.threedi import config

blueprint = blueprints.Blueprint(name=config.BLUEPRINT_NAME,
                                 import_name=__name__,
                                 static_folder='static',
                                 template_folder='templates')

@blueprint.route('/hello')
def hello():
    return 'hello'

@blueprint.route('/wms')
def wms():
    """ Return response according to request. """
    get_parameters = utils.get_parameters()
    request = get_parameters['request'].lower()

    request_handlers = dict(
        getinfo=responses.get_response_for_getinfo,
        getmap=responses.get_response_for_getmap,
        getcapabilities=responses.get_response_for_getcapabilities,
    )
    return request_handlers[request](get_parameters=get_parameters)

@blueprint.route('/data')
def data():
    """ 
    Return data according to request:
    getprofile, gettimeseries, get
    """
    get_parameters = utils.get_parameters()
    request = get_parameters['request'].lower()

    request_handlers = dict(
        getprofile=responses.get_response_for_getprofile,
        gettimeseries=responses.get_response_for_gettimeseries,
        getquantity=responses.get_response_for_getquantity,
        getcontours=responses.get_response_for_getcontours,
    )
    return request_handlers[request](get_parameters=get_parameters)

@blueprint.route('/demo')
def demo():
    layers = os.listdir(config.DATA_DIR)
    return flask.render_template('3di/demo.html',
                                 layers=layers)


@blueprint.route('/leaflet')
def leaflet():
    layers = os.listdir(config.DATA_DIR)
    return flask.render_template('3di/leaflet.html',
                                 layers=layers)


@blueprint.route('/tms/<int:zoom>/<int:x>/<int:y>.png')
def tms(x, y, zoom):
    # Determe bbox from tile indices and zoomlevel
    # Could use tms option for leaflet to inverse y-axis!
    limit = 2 * np.pi * 6378137
    step = limit / 2 ** zoom
    left = step * x - limit / 2
    right = step * (x + 1) - limit / 2
    bottom = limit - step * (y + 1) - limit / 2
    top = limit - step * y - limit / 2
    srs = 'EPSG:3857'
    bbox = ','.join(map(str, [left, bottom, right, top]))
    width, height = '256', '256'  # Standard tile size always?
    #request = 'GetMap'

    get_parameters = utils.get_parameters()
    get_parameters.update(srs=srs,
                          bbox=bbox,
                          width=width,
                          height=height)
    return responses.get_response_for_getmap(get_parameters=get_parameters)
