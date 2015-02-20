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
        'disable_existing_loggers': False,
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
    """Setup logging according to logfile and settings."""
    logfile_path = os.path.join(config.LOG_DIR, logfile_name)
    if hasattr(config, 'LOGGING'):
        logging_dict = config.LOGGING
        logging_dict['handlers']['file']['filename'] = logfile_path
    else:
        logging_dict = _get_logging_dict(logfile_path=logfile_path)
    # create directory if necessary
    try:
        os.makedirs(os.path.dirname(
            logging_dict['handlers']['file']['filename'],
        ))
    except OSError:
        pass  # probably already exists
    # config logging
    logging.config.dictConfig(logging_dict)
