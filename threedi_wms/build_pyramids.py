# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from threedi_wms.threedi import config
from threedi_wms.threedi import tasks
from threedi_wms.threedi import utils

from gislib import rasters

import os
import logging

logger = logging.getLogger(__name__)


def main():
    """
    Check and build pyramids for all missing cache entries for available 3Di
    models.

    layer is Directory name of model.
    """
    logger.info('Build pyramids for all models that do not have pyramids yet')
    for layer in os.listdir(config.DATA_DIR):
        if not os.path.isdir(os.path.join(config.DATA_DIR, layer)):
            continue

        pyramid_path = utils.get_pyramid_path(layer)
        pyramid = rasters.Pyramid(path=pyramid_path,
                                  compression='DEFLATE')
        if not pyramid.has_data():
            logger.info('%s: No pyramid data, going to build.' % layer)
            tasks.make_pyramid.delay(layer)

        monolith_path = os.path.join(config.CACHE_DIR, layer, 'monolith')
        monolith = rasters.Monolith(path=monolith_path, compression='DEFLATE')

        if not monolith.has_data():
            logger.info('%s: No monolith data, going to build.' % layer)
            tasks.make_monolith.delay(layer=layer)
