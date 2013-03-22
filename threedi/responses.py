# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from threedi import config
from threedi import tasks
from threedi import utils

from gislib import raster

from PIL import Image
from netCDF4 import Dataset
from scipy import ndimage
from matplotlib import colors

import numpy as np

import collections
import datetime
import io
import json
import logging
import os

cache = {}


def get_array(container, width, height, bbox, srs, ma=False, **kwargs):
    """
    Return numpy (masked) array.

    Kwargs are not used, but make it possible to pass a wms get parameterlist.
    """
    geometry = raster.DatasetGeometry(
        size=(int(width), int(height)),
        extent=map(float, bbox.split(',')),
    )
    dataset = geometry.to_dataset(
        datatype=container.datatype,
        projection=srs,
    )
    container.warpinto(dataset)
    data = dataset.ReadAsArray()

    if ma:
        return np.ma.array(data,
                           mask=np.equal(data, container.nodatavalue))
    else:  # for readability
        return data


def get_image(masked_array, waves=None):
    """ Return a png image from masked_array. """
    # Hardcode depth limits, until better height data
    normalize = colors.Normalize(vmin=0, vmax=2)
    # Custom color map
    cdict = {
        'red': ((0.0, 170. / 256, 170. / 256),
                (0.5, 65. / 256, 65. / 256),
                (1.0, 4. / 256, 4. / 256)),
        'green': ((0.0, 200. / 256, 200. / 256),
                  (0.5, 120. / 256, 120. / 256),
                  (1.0, 65. / 256, 65. / 256)),
        'blue': ((0.0, 255. / 256, 255. / 256),
                 (0.5, 221. / 256, 221. / 256),
                 (1.0, 176. / 256, 176. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)
    # Apply scaling and colormap
    arr = masked_array
    #import pdb; pdb.set_trace()
    if waves is not None:
        arr += waves
    rgba = colormap(normalize(arr), bytes=True)
    # Make negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0)] = 0

    # Create image
    image = Image.fromarray(rgba)
    buf = io.BytesIO()
    image.save(buf, 'png')
    return buf.getvalue()


def get_water_waves(masked_array, anim_frame):
    """
    Calculate waves from velocity array
    """
    # Animating 'waves'
    y_shape, x_shape = masked_array.shape
    x, y = np.mgrid[0:y_shape, 0:x_shape]
    offset = anim_frame * 0.01

    magnitude = masked_array
    waves = (np.sin(np.pi * 64 / magnitude *
             (offset + x / x_shape + y / y_shape)) * magnitude +
             np.sin(np.pi * 60 / magnitude *
             (offset + y / y_shape)) * magnitude)

    # 'Shade' by convolution
    waves_shade = ndimage.filters.convolve(
        waves,
        np.array([[-.2, -0.5, -0.7, -.5, .3],
                  [-.5, -0.7, -1.5,  .4, .5],
                  [-.7, -1.5,  0.0, 1.5, .7],
                  [-.5, -0.4,  1.5,  .7, .5],
                  [-.3,  0.5,  0.7,  .5, .2]]))

    normalize = colors.Normalize(vmin=0, vmax=24)

    return get_image(masked_array, waves=normalize(waves_shade))


def get_waterlevel(quad_data, waterlevel_data, get_parameters):
    """ Return numpy masked array. """
    projection = raster.get_wkt(get_parameters['srs'])
    geometry = raster.DatasetGeometry(
        extent=map(float, get_parameters['bbox'].split(',')),
        size=(int(get_parameters['width']),
              int(get_parameters['height'])),
    )
    # Get waterlevel from the quad pyramid or monolith
    if quad_data.quad_pyramid.toplevel is None:
        quad_datatype = quad_data.quad_monolith.datatype
        ds_quad = geometry.to_dataset(datatype=quad_datatype)
        ds_quad.SetProjection(projection)
        quad_data.quad_monolith.warpinto(ds_quad)
    else:
        quad_datatype = quad_data.quad_pyramid.datatype
        ds_quad = geometry.to_dataset(datatype=quad_datatype)
        ds_quad.SetProjection(projection)
        quad_data.quad_pyramid.warpinto(ds_quad)
    quad = ds_quad.ReadAsArray()
    waterlevel = waterlevel_data.waterlevel[quad]

    return waterlevel


# Responses for various requests
def get_response_for_getinfo(get_parameters):
    """ Return json with bounds and timesteps. """
    # Read netcdf
    path = utils.get_netcdf_path(layer=get_parameters['layer'])
    with Dataset(path) as dataset:
        v = dataset.variables
        fex, fey = v['FlowElemContour_x'][:], v['FlowElemContour_y'][:]
        timesteps = v['s1'].shape[0]
    netcdf_extent = fex.min(), fey.min(), fex.max(), fey.max()

    # Determine transformed extent
    srs = get_parameters['srs']
    if srs:
        source_projection = raster.RD
        target_projection = srs
        extent = raster.get_transformed_extent(
            extent=netcdf_extent,
            source_projection=source_projection,
            target_projection=target_projection,
        )
    else:
        extent = netcdf_extent

    # Prepare response
    content = json.dumps(dict(bounds=extent,
                              timesteps=timesteps))
    return content, 200, {'content-type': 'application/json'}


def get_response_for_getmap(get_parameters):
    """ Return png image. """
    # Get the quad and waterlevel data objects
    layer = get_parameters['layer']
    time = int(get_parameters['time'])
    try:
        static_data = StaticData.get(layer=layer)
    except ValueError:
        return 'Objects not ready, starting preparation.'
    except raster.LockError:
        return 'Objects not ready, preparation in progress.'
    dynamic_data = DynamicData.get(layer=layer, time=time)

    # Get height
    height_time = datetime.datetime.now()
    height = get_array(container=static_data.pyramid,
                       ma=True,
                       **get_parameters)
    logging.debug('Got height in {} ms.'.format(
        1000 * (datetime.datetime.now() - height_time).total_seconds(),
    ))

    # Get waterlevel
    quad_time = datetime.datetime.now()
    quad = get_array(container=static_data.monolith,
                     ma=False,
                     **get_parameters)
    logging.debug('Got quad in {} ms.'.format(
        1000 * (datetime.datetime.now() - quad_time).total_seconds(),
    ))

    waterlevel = dynamic_data.waterlevel[quad]

    # Combine and return response
    depth = waterlevel - height

    if 'anim_frame' in get_parameters:
        # Add wave animation
        content = get_water_waves(masked_array=depth,
                                  anim_frame=int(get_parameters['anim_frame']))
    else:
        # Direct image
        content = get_image(waterlevel - height)
    return content, 200, {'content-type': 'image/png'}


class StaticData(object):
    """
    Container for static data from the netcdf.
    """
    @classmethod
    def get(cls, layer):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        # Prepare key
        key = collections.namedtuple(
            'StaticDataKey', ['layer'],
        )(layer=layer)

        # Return object
        try:
            return cache[key]
        except KeyError:
            value = cls(layer=layer)
            cache[key] = value
            return value

    def __init__(self, layer):
        """ Init pyramid and monolith, and order creation if necessary. """
        # Initialize pyramid for bathymetry
        pyramid_path = utils.get_pyramid_path(layer)
        pyramid = raster.Pyramid(path=pyramid_path,
                                 compression='DEFLATE')
        # Order building if necessary
        if not pyramid.has_data():
            tasks.make_pyramid.delay(layer)
            raise ValueError('Pyramid not ready yet, task submitted.')
        # If all ok, set pyramid attribute.
        self.pyramid = pyramid

        # Initialize monolith for quad layout
        monolith_path = os.path.join(config.CACHE_DIR, layer, 'monolith')
        monolith = raster.Monolith(path=monolith_path,
                                   memory=True,
                                   compression='DEFLATE')
        # Order building if necessary
        if not monolith.has_data():
            tasks.make_monolith.delay(layer=layer)
            raise ValueError('Monolith not ready yet, task submitted.')
        # If all ok, set monolith attribute.
        self.monolith = monolith


class DynamicData(object):
    """
    Container for only the waterlevel data from the netcdf.
    """
    @classmethod
    def get(cls, layer, time):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        # Prepare key
        key = collections.namedtuple(
            'DynamicDataKey', ['layer', 'time'],
        )(layer=layer, time=time)

        # Return object
        try:
            return cache[key]
        except KeyError:
            value = cls(layer=layer, time=time)
            cache[key] = value
            return value

    def __init__(self, layer, time):
        """ Load data from netcdf. """
        netcdf_path = utils.get_netcdf_path(layer)
        with Dataset(netcdf_path) as dataset:
            waterlevel_variable = dataset.variables['s1']

            # Initialize empty array with one element more than amount of quads
            self.waterlevel = np.ma.array(
                np.empty(waterlevel_variable.shape[1] + 1),
                mask=True,
            )

            # Fill with waterlevel from netcdf
            if time < waterlevel_variable.shape[0]:
                corrected_time = time
            else:
                corrected_time = waterlevel_variable.shape[0] - 1
            self.waterlevel[0:-1] = waterlevel_variable[corrected_time]
