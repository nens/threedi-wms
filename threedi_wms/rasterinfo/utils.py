#!/usr/bin/env python
#
# utils to manage the pyramid

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from gislib.raster import Pyramid
from rasterinfo import config
from osgeo import gdal

import sys
import getopt


def init_pyramid(path=config.PYRAMID_PATH):
    """
    setup Pyramid

    :param path: path to Pyramid defaults to path in config
    """
    return Pyramid(path)


def add_data(path):
    """
    Add data to the Pyramid

    :param path: path to gdal readable raster file
    """
    if not path:
        return "no path specified"

    pyramid = init_pyramid()
    try:
        dataset = gdal.Open(path)
        pyramid.add(dataset)
        return "Data added fine"
    except:
        return "Error"


def usage():
    """
    Usage string for command line
    """
    print("""
    Usage: utils.py <rasterfile1> [<rasterfile2>]

    where <rasterfile?> is the path to a gdal readable raster

    -h, --help  print this message
    """)


def main(argv):
    """
    Add data to Pyramid
    """
    try:
        opts, args = getopt.getopt(argv, "h", ["help"])
    except getopt.GetOptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("h", "--help"):
            usage()
            sys.exit()
    for arg in args:
        print(arg)
        r = add_data(arg)
        print(r)

if __name__ == '__main__':
    main(sys.argv[1:])
