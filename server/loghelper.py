# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging.config
import logging
import os

from server import config


def _get_logging_dict(logfile_path):
    return {
        'disable_existing_loggers': True,
        'version': 1,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(message)s'
            },
            'simple': {
                'format': '%(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'verbose',
                'filename': logfile_path,
                'mode': 'a',
                'maxBytes': 10485760,
                'backupCount': 5,
            },
        },
        'loggers': {
            'root': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
            },
            '': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
            },
        }
    }


def setup_logging(logfile_name):
    """ Setup logging according to logfile and settings. """
    # Get logging dictionary
    logfile_path = os.path.join(config.LOG_DIR, logfile_name)
    logging_dict = _get_logging_dict(logfile_path=logfile_path)
    # Create directory if necessary
    try:
        os.makedirs(os.path.dirname(
            logging_dict['handlers']['file']['filename'],
        ))
    except OSError:
        pass  # Already exists
    # Config logging
    logging.config.dictConfig(logging_dict)
