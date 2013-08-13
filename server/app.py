# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from server import blueprints
from server import loghelper

import flask

# App
app = flask.Flask(__name__)

# Register the blueprints
for blueprint in blueprints.get_blueprints():
    url_prefix = '/' + blueprint.name
    app.register_blueprint(blueprint, url_prefix=url_prefix)

# Setup logging
loghelper.setup_logging(logfile_name='server.log')

# Main
def run():
    app.run(host='0.0.0.0', debug=True)
