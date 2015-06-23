# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging

from gislib import rasters

from netCDF4 import Dataset
from scipy import spatial
from osgeo import gdal

import numpy as np

logger = logging.getLogger(__name__)


def get_dataset(path, projection=None):
    """ Return a gdaldataset containing the quad positions.

    projection is None or 22234 (int) or something"""
    # Load data
    with Dataset(path) as dataset:
        v = dataset.variables
        try:
            nr_of_2d_elements = dataset.nFlowElem2d
        except AttributeError:
            # No dataset.nFlowElem2d. Assume all variables are 2D.
            fex, fey = v['FlowElemContour_x'][:], v['FlowElemContour_y'][:]
            fcx, fcy = v['FlowElem_xcc'][:], v['FlowElem_ycc'][:]
        else:
            # This dataset contains nFlowElem2d. Using it here to only get
            # 2D elements.
            fex = v['FlowElemContour_x'][:nr_of_2d_elements]
            fey = v['FlowElemContour_y'][:nr_of_2d_elements]
            fcx = v['FlowElem_xcc'][:nr_of_2d_elements]
            fcy = v['FlowElem_ycc'][:nr_of_2d_elements]
        # For boezemstelsel Delfland only
        # fex, fey = v['FlowElemContour_x'][:1], v['FlowElemContour_y'][:1]
        # fcx, fcy = v['FlowElem_xcc'][:1], v['FlowElem_ycc'][:1]
    x1, y1, x2, y2 = fex.min(1), fey.min(1), fex.max(1), fey.max(1)

    # Set convenient arrays
    widths = x2 - x1
    heights = y2 - y1
    areas = widths * heights
    centers = np.array([fcx, fcy]).T
    extent = (x1.min(), y1.min(),
              x2.max(), y2.max())

    # Determine the grid based on the smallest quads.
    width = int(round((extent[2] - extent[0]) / widths.min()))
    height = int(round((extent[3] - extent[1]) / heights.min()))
    geometry = rasters.DatasetGeometry(
        extent=extent, size=(width, height),
    )
    gridpoints = geometry.gridpoints()
    quad_grid = np.ma.array(np.empty((height, width)), mask=True)

    # Loop quads grouped by area
    for area in np.unique(areas):

        index = (areas == area)

        # Prepare an array with indices to current quads
        count = index.sum()
        quad_index = np.ma.array(np.empty(count + 1), mask=True)
        quad_index[:count] = np.arange(index.size)[index]

        # Construct and query a nearest-neighbour interpolator
        upper = np.sqrt(area) / 2
        logger.debug('Adding quads of area {} to dataset.'.format(area))
        data_index = spatial.cKDTree(centers[index]).query(
            gridpoints, p=np.inf, distance_upper_bound=upper,
        )[1]

        # Add to result
        quad_grid = np.ma.sum([
            quad_grid,
            quad_index[data_index].reshape(height, width)
        ], axis=0)

    dataset = geometry.to_dataset(projection=projection,
                                  datatype=gdal.GDT_UInt32)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(areas.size)
    band.WriteArray(quad_grid.filled(areas.size))
    return dataset
