# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

import server
import flask

from threedi import responses
from threedi import config


blueprint = server.Blueprint(name=config.BLUEPRINT_NAME,
                             import_name=__name__,
                             static_folder='static',
                             template_folder='templates')


def get_dataset_list():
    datasets = []
    for basedir, dirnames, filenames in os.walk(config.DATA_DIR):
        for filename in filenames:
            if filename.endswith('.nc'):
                datasets.append(dict(
                    filepath=os.path.join(
                        basedir, filename,
                    ).replace(config.DATA_DIR + '/', ''),
                    filename=filename,
                ))
    return datasets


@blueprint.route('/wms')
def wms():
    
    """ Return response according to request. """
    get_parameters = server.utils.get_parameters()
    request = get_parameters['request'].lower()

    request_handlers = dict(
        getinfo=responses.get_response_for_getinfo,
        getmap=responses.get_response_for_getmap,
        prepare=responses.get_response_for_prepare,
    )
    return request_handlers[request](get_parameters=get_parameters)


@blueprint.route('/demo')
def demo():
    from server import tasks
    tasks.build_pyramid.delay()
    return flask.render_template('3di/demo.html', 
                                 datasets=get_dataset_list())
