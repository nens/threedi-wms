    # -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from server import blueprints
from server import loghelper
from server.messages import MessageData
from server import config

import flask
from flask.ext.cache import Cache
from raven.contrib.flask import Sentry


def build_app(sub_port=5558, **kwargs):
    """App is already global and existing"""
    global app
    global message_data
    global cache

    print("Starting threedi-wms...")
    # Setup logging
    loghelper.setup_logging(logfile_name='server.log')
    # Using print because I don't see logging output on screen while running manually
    # print("request port: %d (server should process requests on this port)" % req_port)
    print("subscription port: %d (server should publish on this port)" % sub_port)

    # App
    app = flask.Flask(__name__)
    if config.USE_CACHE:
        cache_config = {
            'CACHE_TYPE': 'redis', 
            'CACHE_KEY_PREFIX': config.CACHE_PREFIX,
            'CACHE_REDIS_DB': 3,
            }
    else:
        cache_config = {
            'CACHE_TYPE': 'null', }
    cache = Cache(app, config=cache_config)

    if hasattr(config, 'SENTRY_DSN'):
        app.config['SENTRY_DSN'] = (config.SENTRY_DSN)
        sentry = Sentry(app)

    # this one is global because we only have one event loop that receives messages
    message_data = MessageData(sub_port=sub_port)
    # stop listenin when we tear down the app
    # flask.appcontext_tearing_down.connect?

    # Register the blueprints
    for blueprint in blueprints.get_blueprints():
        url_prefix = '/' + blueprint.name
        app.register_blueprint(blueprint, url_prefix=url_prefix)

    print("ready to rock and roll!")

    return app


# Main
def run():
    app = build_app(sub_port=5558)
    app.run(host='0.0.0.0', debug=True)


