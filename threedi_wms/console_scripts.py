# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from threedi_wms.threedi import tasks

import os
import logging
import sys


logger = logging.getLogger(__name__)


def process_threedi_result():
    """
    Dispatch celery task process_threedi_result
    """
    logger.info('Dispatching celery task process_threedi_result...')
    try:
        input_folder, output_folder = sys.argv[1], sys.argv[2]
    except:
        exit("Provide <input folder> <output_folder>")
    logger.info('Input folder: %s' % input_folder)
    logger.info('Output folder: %s' % output_folder)
    tasks.process_threedi_result.delay(input_folder, output_folder)


def dump_memory():
    logger.info('Dumping memory...')
    try:
        output_folder = sys.argv[1]
    except:
        exit("Provide <output_folder>")
    logger.info('Output folder: %s' % output_folder)
    tasks.dump_memory.delay(output_folder)
