# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging
import logging.config
import time

from server import blueprints
from server.messages import MessageData
from server import config
from server import status
from server import utils

import flask
from flask.ext.cache import Cache
from raven.contrib.flask import Sentry

# these two lines are needed for celery; the real app cache is set in the
# build_app function below
_app = flask.Flask(__name__)
cache = Cache(_app, config={'CACHE_TYPE': 'null', })

reporter = status.StateReporter()

logger = logging.getLogger('')


def build_app(sub_port=5558, **kwargs):
    """Build the flask app."""

    global app
    global message_data
    global cache

    logger.info("Starting threedi-wms...")
    logger.info("Subscription port: %d (server should publish on this port)." %
                sub_port)

    # use the correct subgrid_id
    subgrid_id = utils.fetch_subgrid_id()
    logger.info(
        "Got subgrid_id: %s." % subgrid_id, extra={'subgrid_id': subgrid_id})

    app = flask.Flask(__name__)
    if config.USE_CACHE:
        cache_config = {
            'CACHE_TYPE': 'redis',
            'CACHE_KEY_PREFIX': subgrid_id,
            'CACHE_REDIS_HOST': config.REDIS_HOST_CACHE,
            'CACHE_REDIS_PORT': config.REDIS_PORT,
            'CACHE_REDIS_DB': config.REDIS_DB_THREEDI_WMS_CACHE,
        }
    else:
        cache_config = {'CACHE_TYPE': 'null', }
    cache = Cache(app, config=cache_config)

    # reset state variables
    logger.info("Reset wms state variables in redis.")
    reporter.reset_all()

    # setup sentry
    if hasattr(config, 'SENTRY_DSN'):
        app.config['SENTRY_DSN'] = config.SENTRY_DSN
        Sentry(app, dsn=config.SENTRY_DSN, logging=True, level=logging.ERROR)

    # this one is global because we only have one event loop that receives
    # messages
    message_data = MessageData(sub_port=sub_port)

    # register the blueprints
    for blueprint in blueprints.get_blueprints():
        url_prefix = '/' + blueprint.name
        app.register_blueprint(blueprint, url_prefix=url_prefix)

    logger.info("Ready to rock and roll!", extra={'subgrid_id': subgrid_id})
    return app


def run():
    app = build_app(sub_port=5558)
    app.run(host='0.0.0.0', debug=True)
