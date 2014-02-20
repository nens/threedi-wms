# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from server import blueprints
from server import loghelper
from server.messages import MessageData

import flask


def build_app(req_port=5556, sub_port=5558):
    """App is already global and existing"""
    global app
    global message_data

    # App
    app = flask.Flask(__name__)

    # this one is global because we only have one event loop that receives messages
    message_data = MessageData(req_port=5556, sub_port=5558)

    # Register the blueprints
    for blueprint in blueprints.get_blueprints():
        url_prefix = '/' + blueprint.name
        app.register_blueprint(blueprint, url_prefix=url_prefix)

    # Setup logging
    loghelper.setup_logging(logfile_name='server.log')
    print("request port: %d (server should process requests on this port)" % req_port)
    print("subscription port: %d (server should publish on this port)" % sub_port)

    return app

# Main
def run():
    build_app(req_port=5556, sub_port=5558).run(host='0.0.0.0', debug=True)
