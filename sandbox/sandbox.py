# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import sys

from osgeo import gdal

from gislib import raster
from server import loghelper

loghelper.setup_logging(logfile_name='sandbox.log')



def main():
    pyramid = raster.Pyramid(path=sys.argv[1])
    pyramid.add(gdal.Open(sys.argv[2]))
