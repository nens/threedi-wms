# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from threedi_wms.threedi import config
from threedi_wms.threedi import tasks
from threedi_wms.threedi import utils

from gislib import rasters
from gislib import vectors as vector
from gislib import utils as gislib_utils

from PIL import Image
from netCDF4 import Dataset
from netCDF4 import num2date
from matplotlib import cm
from matplotlib import colors
from scipy import ndimage

import numpy as np
import ogr
import redis

import datetime
import io
import json
import logging
import math
import os
import time as _time  # stop watch
import osr

from server.app import cache
from server import config as redis_config
from server import utils as server_utils

ogr.UseExceptions()

logger = logging.getLogger('')

rc_node = redis.Redis(host=redis_config.REDIS_HOST,
                      port=redis_config.REDIS_PORT,
                      db=redis_config.REDIS_NODE_MAPPING_DB)

rc_state = redis.Redis(host=redis_config.REDIS_HOST,
                       port=redis_config.REDIS_PORT,
                       db=redis_config.REDIS_STATE_DB)

PANDAS_VARS = ['pumps', 'weirs', 'orifices', 'culverts']
KNOWN_VARS = ['pumps', 'weirs', 'orifices', 'culverts', 'unorm', 'q']

CSV_HEADER = ['datetime', 'value', 'unit', 'object_id', 'object_type']


def rgba2image(rgba):
    """ return imagedata. """
    image = Image.fromarray(rgba)

    buf = io.BytesIO()
    image.save(buf, 'png')
    return buf.getvalue(), image


def get_depth_image(masked_array, hmin=0, hmax=2):
    """ Return a png image from masked_array. """
    # Hardcode depth limits, until better height data
    normalize = colors.Normalize(vmin=hmin, vmax=hmax)
    normalized_arr = normalize(masked_array)
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
                 (1.0, 146. / 256, 146. / 256)),
        'alpha': ((0.0, 64. / 256, 64. / 256),
                  (0.1, 128. / 256, 128. / 256),
                  (0.5, 256. / 256, 256. / 256),
                  (1.0, 256. / 256, 256. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)
    # If matplotlib does not support alpha and you want it anyway:
    # Use red as alpha, then overwrite the alpha channel
    cdict2 = {  # some versions of matplotlib do not have alpha
        'green': ((0.0, 200. / 256, 200. / 256),
                  (1.0, 65. / 256, 65. / 256)),
        'blue': ((0.0, 255. / 256, 255. / 256),
                 (1.0, 146. / 256, 146. / 256)),
        # alpha!!
        'red': ((0.0, 0. / 256, 0. / 256),
                (0.03, 32. / 256, 32. / 256),
                (0.07, 64. / 256, 64. / 256),
                (0.2, 128. / 256, 128. / 256),
                (0.5, 256. / 256, 256. / 256),
                (1.0, 256. / 256, 256. / 256)),
    }
    colormap2 = colors.LinearSegmentedColormap('something', cdict2, N=1024)
    rgba2 = colormap2(normalized_arr, bytes=True)
    rgba[..., 3] = rgba2[..., 0]

    # Make very small/negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0.01)] = 0
    rgba[masked_array.mask, 3] = 0

    return rgba2image(rgba=rgba)


def get_soil_image(masked_array, hmin=0, hmax=7):
    """ Return a png image from masked_array. """
    colormap = colors.ListedColormap([
        '#FFFFFF',  # empty, starts at 1
        '#F2BF24',  # Veengrond met veraarde bovengrond
        '#8F793C',  # Veengrond met veraarde bovengrond, zand
        '#63160D',  # Veengrond met kleidek
        '#14208C',  # Veengrond met kleidek op zand
        '#9943E6',  # Veengrond met zanddek op zand
        '#29FA11',  # Veengrond op ongerijpte klei
        '#F2F5A9',  # Stuifzand
        '#09701A',  # Podzol (Leemarm, fijn zand)
        '#309AE6',  # Podzol (zwak lemig, fijn zand)
        '#1FED86',  # Podzol (zwak lemig, fijn zand op grof zand
        '#A3E014',  # Podzol (lemig keileem)
        '#363154',  # Enkeerd (zwak lemig, fijn zand)
        '#F7C6EA',  # Beekeerd (lemig fijn zand)
        '#25F7F0',  # Podzol (grof zand)
        '#C75B63',  # Zavel
        '#3613E8',  # Lichte klei
        '#DB0E07',  # Zware klei
        '#D1680D',  # Klei op veen
        '#275C51',  # Klei op zand
        '#8A084B',  # Klei op grof zand
        '#886A08',  # Leem
        ])
    bounds = range(22)
    normalize = colors.BoundaryNorm(bounds, colormap.N)
    normalized_arr = normalize(masked_array)
    # Custom color map
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)

    # Make very small/negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0.01)] = 0
    rgba[masked_array.mask, 3] = 0

    return rgba2image(rgba=rgba)


def get_crop_image(masked_array, hmin=0, hmax=7):
    """ Return a png image from masked_array. """
    colormap = colors.ListedColormap([
        '#FFFFFF',  # empty, starts at 1
        '#76CD2F',  # ' . grass ', & rgb(118, 205, 47)
        '#F1C40F',  # ' . corn ', & rgb(241, 196, 15)
        '#F39C12',  # ' . potatoes ', & rgb(243, 156, 18)
        '#CD2FAD',  # ' . sugarbeet ', & rgb(205, 47, 173)
        '#FDFF41',  # ' . grain ', & rgb(253, 255, 65)
        '#2ECC71',  # ' . miscellaneous ', & rgb(46, 204, 113)
        '#886A08',  # ' . non-arable land', & rgb(136, 106, 8)
        '#0489B1',  # ' . greenhouse area', & rgb(4, 137, 177)
        '#173B0B',  # ' . orchard ', & rgb(23, 59, 11)
        '#B45F04',  # '. bulbous plants ', & rgb(180, 95, 4)
        '#2F5A00',  # '. foliage forest ', & rgb (47, 90, 0)
        '#38610B',  # '. pine forest ', & rgb(56, 97, 11)
        '#16A085',  # '. nature ', & rgb(22, 160, 133)
        '#61380B',  # '. fallow ', & rgb(97, 56, 11)
        '#16A085',  # '. vegetables ', & rgb(22, 160, 133)
        '#9B59B6',  # '. flowers '/) rgb(155, 89, 182)
        ])
    bounds = range(17)
    normalize = colors.BoundaryNorm(bounds, colormap.N)
    normalized_arr = normalize(masked_array)
    # Custom color map
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)

    # Make very small/negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0.01)] = 0
    rgba[masked_array.mask, 3] = 0

    return rgba2image(rgba=rgba)


def get_arrival_image(masked_array, hmin=0, hmax=7):
    """Return a png image from masked_array."""
    normalize = colors.Normalize(vmin=hmin, vmax=hmax)
    normalized_arr = normalize(masked_array)
    # Custom color map
    cdict = {
        'red': ((0.0, 255. / 256, 255. / 256),
                (1.0, 255. / 256, 255. / 256)),
        'green': ((0.0, 0. / 256, 0. / 256),
                  (1.0, 255. / 256, 255. / 256)),
        'blue': ((0.0, 0. / 256, 0. / 256),
                 (1.0, 0. / 256, 0. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)

    # Make very small/negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0.01)] = 0
    rgba[masked_array.mask, 3] = 0

    return rgba2image(rgba=rgba)


def get_bathymetry_image(masked_array, limits):
    """Return imagedata."""
    normalize = colors.Normalize(vmin=limits[0], vmax=limits[1])
    normalized_arr = normalize(masked_array)
    # Custom color map
    cdict = {
        'red': ((0.0, 253. / 256, 253. / 256),
                (0.14, 32. / 256, 32. / 256),
                (0.28, 124. / 256, 124. / 256),
                (0.43, 255. / 256, 255. / 256),
                (0.57, 133. / 256, 133. / 256),
                (0.71, 107. / 256, 107. / 256),
                (0.85, 150. / 256, 150. / 256),
                (1.0, 230. / 256, 230. / 256)),
        'green': ((0.0, 255. / 256, 255. / 256),
                  (0.14, 148. / 256, 148. / 256),
                  (0.28, 160. / 256, 160. / 256),
                  (0.43, 78. / 256, 78. / 256),
                  (0.57, 42. / 256, 42. / 256),
                  (0.71, 64. / 256, 64. / 256),
                  (0.85, 150. / 256, 150. / 256),
                  (1.0, 230. / 256, 230. / 256)),
        'blue': ((0.0, 92. / 256, 92. / 256),
                 (0.14, 29. / 256, 29. / 256),
                 (0.28, 46. / 256, 46. / 256),
                 (0.43, 0. / 256, 0. / 256),
                 (0.57, 2. / 256, 2. / 256),
                 (0.71, 46. / 256, 46. / 256),
                 (0.85, 150. / 256, 150. / 256),
                 (1.0, 230. / 256, 230. / 256)),
        'alpha': ((0.0, 256. / 256, 256. / 256),
                  (1.0, 256. / 256, 256. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)

    return rgba2image(rgba=rgba)


def get_color_image(masked_array, color_a=None, color_b=None, vmin=0, vmax=1):
    """Return imagedata in a sort of rainbow."""
    if color_a is None:
        # default: magenta
        color_a = (256, 50, 256)
    if color_b is None:
        # default: green
        color_b = (50, 256, 50)
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    normalized_arr = normalize(masked_array)
    # Custom color map
    cdict = {
        'red': ((0.0, color_a[0] / 256., color_a[0] / 256.),
                (0.5, 256. / 256., 256. / 256.),
                (1.0, color_b[0] / 256., color_b[0] / 256.)),
        'green': ((0.0, color_a[1] / 256, color_a[1] / 256),
                  (0.5, 256. / 256, 256. / 256),
                  (1.0, color_b[1] / 256, color_b[1] / 256)),
        'blue': ((0.0, color_a[2] / 256, color_a[2] / 256),
                 (0.5, 256. / 256, 256. / 256),
                 (1.0, color_b[2] / 256, color_b[2] / 256)),
        'alpha': ((0.0, 256. / 256, 256. / 256),
                  (1.0, 256. / 256, 256. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)
    # Apply scaling and colormap

    rgba = colormap(normalized_arr, bytes=True)

    return rgba2image(rgba=rgba)


def get_grid_image(masked_array):
    """Return imagedata."""
    a, b = -1, 8
    kernel = np.array([[a,  a, a],
                       [a,  b, a],
                       [a,  a, a]])
    data = ndimage.filters.convolve(masked_array, kernel)
    normalize = colors.Normalize()
    rgba = np.zeros(data.shape + (4,), dtype=np.uint8)
    index = np.ma.greater(normalize(data), 0.5)
    rgba[index] = (255, 0, 0, 255)
    rgba[~index] = (255, 0, 0, 0)
    return rgba2image(rgba=rgba)


def get_quad_grid_image(masked_array):
    """Return imagedata."""
    normalize = colors.Normalize()
    colormap = cm.Set2
    rgba = colormap(normalize(masked_array), bytes=True)
    return rgba2image(rgba=rgba)


def get_velocity_image(masked_array, vmin=0, vmax=1.):
    """Return imagedata."""
    # Custom color map
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    normalized_arr = normalize(masked_array)
    cdict = {
        'green': ((0.0, 170. / 256, 170. / 256),
                  (0.5, 65. / 256, 65. / 256),
                  (1.0, 4. / 256, 4. / 256)),
        'blue': ((0.0, 200. / 256, 200. / 256),
                 (0.5, 120. / 256, 120. / 256),
                 (1.0, 65. / 256, 65. / 256)),
        'red': ((0.0, 255. / 256, 255. / 256),
                (0.5, 221. / 256, 221. / 256),
                (1.0, 146. / 256, 146. / 256)),
        'alpha': ((0.0, 0. / 256, 0. / 256),
                  (0.1, 64. / 256, 64. / 256),
                  (0.4, 128. / 256, 128. / 256),
                  (0.5, 256. / 256, 256. / 256),
                  (1.0, 256. / 256, 256. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)

    rgba = colormap(normalized_arr, bytes=True)

    cdict2 = {  # some versions of matplotlib do not have alpha
        'green': ((0.0, 200. / 256, 200. / 256),
                  (1.0, 65. / 256, 65. / 256)),
        'blue': ((0.0, 255. / 256, 255. / 256),
                 (1.0, 146. / 256, 146. / 256)),
        # alpha!!
        'red': ((0.0, 0. / 256, 0. / 256),
                (0.1, 64. / 256, 64. / 256),
                (0.4, 128. / 256, 128. / 256),
                (0.5, 256. / 256, 256. / 256),
                (1.0, 256. / 256, 256. / 256)),
    }
    colormap2 = colors.LinearSegmentedColormap('something', cdict2, N=1024)
    rgba2 = colormap2(normalized_arr, bytes=True)
    rgba[..., 3] = rgba2[..., 0]

    # Only show velocities that matter.
    rgba[..., 3][np.ma.less_equal(masked_array, 0.01)] = 0.

    return rgba2image(rgba=rgba)


def get_groundwater_image(masked_array, vmin=0, vmax=3.):
    """Return imagedata."""
    # Custom color map
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)
    normalized_arr = normalize(masked_array)
    cdict = {
        'red': ((0.0, 118. / 256, 118. / 256),
                (0.4, 64. / 256, 64. / 256),
                (0.7, 28. / 256, 28. / 256),
                (0.85, 78. / 256, 78. / 256),
                (1.0, 124. / 256, 124. / 256)),
        'green': ((0.0, 225. / 256, 225. / 256),
                  (0.4, 164. / 256, 164. / 256),
                  (0.7, 58. / 256, 58. / 256),
                  (0.85, 80. / 256, 80. / 256),
                  (1.0, 80. / 256, 80. / 256)),
        'blue': ((0.0, 113. / 256, 113. / 256),
                 (0.4, 30. / 256, 30. / 256),
                 (0.7, 18. / 256, 18. / 256),
                 (0.85, 11. / 256, 11. / 256),
                 (1.0, 4. / 256, 4. / 256)),
    }
    colormap = colors.LinearSegmentedColormap('something', cdict, N=1024)

    rgba = colormap(normalized_arr, bytes=True)

    cdict2 = {  # some versions of matplotlib do not have alpha
        'green': ((0.0, 200. / 256, 200. / 256),
                  (1.0, 65. / 256, 65. / 256)),
        'blue': ((0.0, 255. / 256, 255. / 256),
                 (1.0, 146. / 256, 146. / 256)),
        # alpha!!
        'red': ((0.0, 224. / 256, 224. / 256),
                (1.0, 224. / 256, 224. / 256)),
    }
    colormap2 = colors.LinearSegmentedColormap('something', cdict2, N=1024)
    rgba2 = colormap2(normalized_arr, bytes=True)
    rgba[..., 3] = rgba2[..., 0]

    # A trick to filter out pixels outside the model, see messages.
    rgba[..., 3][np.ma.less_equal(masked_array, vmin)] = 0.

    rgba[..., 3][masked_array.mask == True] = 0.

    return rgba2image(rgba=rgba)


def get_data(container, ma=False, **get_parameters):
    """
    Return numpy (masked) array from container
    """
    start = datetime.datetime.now()
    # Derive properties from get_paramaters

    size = (int(get_parameters['width']),
            int(get_parameters['height']))
    extent = map(float, get_parameters['bbox'].split(','))
    srs = get_parameters['srs']

    # Create dataset
    geometry = rasters.DatasetGeometry(size=size, extent=extent)
    dataset = geometry.to_dataset(
        datatype=container.datatype,
        projection=srs,
    )
    if container.nodatavalue is not None:
        band = dataset.GetRasterBand(1)
        band_array = band.ReadAsArray()
        band_array.fill(container.nodatavalue)
        band.WriteArray(band_array)
    container.warpinto(dataset)
    array = dataset.ReadAsArray()

    # Return array or masked array
    time = 1000 * (datetime.datetime.now() - start).total_seconds()
    if ma:
        data = np.ma.array(array,
                           mask=np.equal(array, container.nodatavalue))
    else:
        data = array
    return data, time


def show_error_img():
    """
    Show error image when when no map layer is available.
    """
    img = Image.open(os.path.join(config.STATIC_DIR, 'maperror.png'))
    buf = io.BytesIO()
    img.save(buf, 'png')
    return buf.getvalue(), img


# Responses for various requests
@cache.memoize(timeout=5)
def get_response_for_getmap(get_parameters):
    """ Return png image.

    Available modes:
    depth
    flood
    bathymetry
    velocity
    grid  : uses (old) pyramids
    quad_grid : what's this?
    sg  : ground water (messages only)
    sg_abs : ground water absolute value
    infiltration (messages only)
    interception (messages only)
    soil (messages only)
    crop (messages only)
    maxdepth -> only when grids.nc is used (messages only)
    arrival -> only when grids.nc is used (messages only)
    """
    # No global import, celery doesn't want this.
    from server.app import message_data

    # Get the quad and waterlevel data objects
    layer_parameter = get_parameters['layers']
    if ':' in layer_parameter:
        layer, mode = layer_parameter.split(':')
    else:
        layer, mode = layer_parameter, 'depth'

    if get_parameters.get('messages', 'false') == 'true':
        use_messages = True
    else:
        use_messages = False
    if mode == 'maxdepth' or mode == 'arrival':
        use_messages = True  # Always use_messages = True

    hmax = get_parameters.get('hmax', 2.0)
    time = int(get_parameters.get('time', 0))

    # Check if messages data is ready. If not: fall back to netcdf/pyramid
    # method.
    if mode == 'maxdepth' or mode == 'arrival':
        required_message_vars = []
        messaging_required = True  # Is it required to use the message method?
    elif mode == 'soil':
        required_message_vars = ['soiltype', ]
        messaging_required = True
    elif mode == 'crop':
        required_message_vars = ['croptype', ]
        messaging_required = True
    elif mode == 'infiltration':
        required_message_vars = ['infiltrationrate', ]
        messaging_required = True
    elif mode == 'interception':
        required_message_vars = ['maxinterception', ]
        messaging_required = True
    else:
        required_message_vars = [
            'dxp', 'wkt', 'quad_grid_dps_mask', 'quad_grid', 's1', 'x1p',
            'y1p', 'jmaxk', 'nodm', 'nodn', 'dyp', 'nodk', 'vol1', 'imax',
            'dsnop', 'imaxk', 'y0p', 'dps', 'jmax', 'x0p']
        messaging_required = False
    if not set(required_message_vars).issubset(set(message_data.grid.keys())):
        missing_vars = (set(required_message_vars) -
                        set(message_data.grid.keys()))
        if messaging_required:
            logger.error('Required vars not available in message_data (mode: '
                         '%s, missing: %r)' % (mode, str(missing_vars)))
            # We cannot do anything for you...
            content, img = show_error_img()

            return content, 200, {
                'content-type': 'image/png',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET'}
        else:
            logger.debug(
                'Not all vars available yet in message_data (missing: %r)'
                ', falling back to netcdf.' % str(missing_vars))
            use_messages = False

    if (use_messages and not messaging_required and
            not message_data.interpolation_ready):

        logger.debug('Interpolation not ready in message_data'
                     ', falling back to netcdf.')
        use_messages = False

    # Pyramid + monolith, when not using messages
    if not use_messages:
        if not os.path.exists(utils.get_netcdf_path(layer=layer)):
            return (
                'Objects not ready, start simulation first [%s not found]' %
                utils.get_netcdf_path(layer=layer))
        try:
            static_data = StaticData.get(layer=layer, reload=False)
        except ValueError:
            return 'Objects not ready, starting preparation.'
        except rasters.LockError:
            return 'Objects not ready, preparation in progress.'

        if mode in ['depth', 'grid', 'flood', 'velocity', 'quad_grid']:
            # lookup quads in target coordinate system
            quads, ms = get_data(container=static_data.monolith,
                                 ma=True, **get_parameters)
            logger.debug('Got quads in {} ms.'.format(ms))

    if mode in ['depth', 'bathymetry', 'flood', 'velocity']:
        # lookup bathymetry in target coordiante system
        ms = 0
        if use_messages and mode == 'bathymetry':
            container = message_data.get('dps', **get_parameters)
            dps, ms = get_data(container=container,
                               ma=True, **get_parameters)
            bathymetry = -dps
        if not use_messages:
            bathymetry, ms = get_data(container=static_data.pyramid,
                                      ma=True, **get_parameters)
        logger.debug('Got bathymetry in {} ms.'.format(ms))

    # The velocity layer has the depth layer beneath it
    if mode == 'depth':
        if use_messages:
            # TODO: cleanup bathymetry. Best do substraction before
            # interpolation
            container = message_data.get(
                "waterlevel", **get_parameters)
            waterlevel, ms = get_data(container, ma=True, **get_parameters)
            depth = waterlevel
        else:
            dynamic_data = DynamicData.get(
                layer=layer, time=time)
            waterlevel = dynamic_data.waterlevel[quads]
            depth = waterlevel - bathymetry

        # Direct image
        content, img = get_depth_image(
            masked_array=depth,
            hmax=hmax)
    elif mode == 'flood':
        # time is actually the sequence number of the flood
        hmax = get_parameters.get('hmax', 2.0)
        time = int(get_parameters['time'])
        dynamic_data = DynamicData.get(
            layer=layer, time=time, variable='floodfill',
            netcdf_path=utils.get_netcdf_path_flood(layer)
        )
        waterlevel = dynamic_data.waterlevel[quads]
        depth = waterlevel - bathymetry

        # Direct image
        content, img = get_depth_image(masked_array=depth,
                                       hmax=hmax)
    elif mode == 'bathymetry':
        logger.debug('bathymetry min, max %r %r' % (
            np.amin(bathymetry), np.amax(bathymetry)))
        limits = map(float, get_parameters['limits'].split(','))
        content, img = get_bathymetry_image(masked_array=bathymetry,
                                            limits=limits)
    elif mode == 'grid':
        content, img = get_grid_image(masked_array=quads)
    elif mode == 'quad_grid':
        content, img = get_quad_grid_image(masked_array=quads)
    elif mode == 'velocity':
        container = message_data.get("uc", **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_velocity_image(masked_array=u)
    elif mode == 'sg':  # ground water, only with use_messages
        container = message_data.get("sg", **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_groundwater_image(masked_array=u)
    elif mode == 'sg_abs':  # ground water, only with use_messages
        container = message_data.get("sg_abs", **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)
        # we use u for visualization only. we want get_groundwater_image
        # inverted.
        u = -u
        vmin = np.amin(u)
        vmax = np.amax(u) + 1  # make it visually more stable
        logger.info("ground control to mayor tom")
        logger.info("%f, %f" % (vmin, vmax))
        content, img = get_groundwater_image(
            masked_array=u, vmin=vmin, vmax=vmax)
    elif mode == 'infiltration':
        container = message_data.get("infiltration", **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_color_image(
            masked_array=u, color_a=(256, 50, 256), color_b=(50, 256, 50),
            vmin=0, vmax=500)
    elif mode == 'interception':
        container = message_data.get("interception", **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_color_image(
            masked_array=u, color_a=(50, 256, 256), color_b=(256, 50, 50),
            vmin=0, vmax=0.020)
    elif mode == 'soil':
        container = message_data.get("soil", **get_parameters)
        if container is None:
            logger.info('WMS not ready for mode: {}'.format(mode))
            content = ''
        else:
            u, ms = get_data(container, ma=True, **get_parameters)
            # funky value coming back from get_data
            # with no_data_value 1410065408 not being picked up by mask
            fix_mask = u == 1410065408
            u.mask = fix_mask

            content, img = get_soil_image(masked_array=u, hmax=22)
    elif mode == 'crop':
        container = message_data.get("crop", **get_parameters)
        if container is None:
            logger.info('WMS not ready for mode: {}'.format(mode))
            content = ''
        else:
            u, ms = get_data(container, ma=True, **get_parameters)
            # funky value coming back from get_data
            # with no_data_value 1410065408 not being picked up by mask
            fix_mask = u == 1410065408
            u.mask = fix_mask

            content, img = get_crop_image(masked_array=u, hmax=16)
    elif mode == 'maxdepth':
        container = message_data.get("maxdepth", from_disk=True,
                                     **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_depth_image(masked_array=u, hmax=hmax)

    elif mode == 'arrival':
        container = message_data.get(
            "arrival", from_disk=True, **get_parameters)
        u, ms = get_data(container, ma=True, **get_parameters)

        content, img = get_arrival_image(masked_array=u, hmax=7)
    else:
        logger.error('Unsupported map requested: %s' % mode)
        content, img = show_error_img()

    return content, 200, {
        'content-type': 'image/png',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getinfo(get_parameters):
    """Return json with bounds and timesteps.

    With attempt to make it work with "messages" as well.
    """
    from server.app import message_data
    if get_parameters.get('messages', 'false') == 'true':
        use_messages = True
    else:
        use_messages = False

    if use_messages:
        container = message_data.get('dps', **get_parameters)
        dps, ms = get_data(container=container,
                           ma=True, **get_parameters)
        bathymetry = -dps
        limits = float(bathymetry.min()), float(bathymetry.max())

        content = json.dumps(dict(limits=limits))
    else:
        # Read netcdf
        path = utils.get_netcdf_path(layer=get_parameters['layers'])
        with Dataset(path) as dataset:
            v = dataset.variables
            fex, fey = v['FlowElemContour_x'][:], v['FlowElemContour_y'][:]
            timesteps = v['s1'].shape[0]
            bathymetry = v['bath'][0, :]

        limits = bathymetry.min(), bathymetry.max()
        netcdf_extent = fex.min(), fey.min(), fex.max(), fey.max()

        # Determine transformed extent
        srs = get_parameters['srs']
        if srs:
            # Read projection from bathymetry file, defaults to RD.
            bathy_path = utils.get_bathymetry_path(
                layer=get_parameters['layers'])
            # It defaults to Rijksdriehoek RD
            source_projection = utils.get_bathymetry_srs(bathy_path)

            logger.info('Source projection: %r' % source_projection)
            target_projection = srs
            extent = gislib_utils.get_transformed_extent(
                extent=netcdf_extent,
                source_projection=source_projection,
                target_projection=target_projection,
            )
        else:
            logger.warning('No srs data available.')
            extent = netcdf_extent

        # Prepare response
        content = json.dumps(dict(bounds=extent,
                                  limits=limits,
                                  timesteps=timesteps))
    return content, 200, {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def wgs_to_rd(lon, lat):
    """
    Transform coordinates in lon/lat to x, y in rd.
    """
    # Spatial Reference System
    inputEPSG = 4326
    outputEPSG = 28992

    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)

    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # transform point
    point.Transform(coordTransform)
    return point.GetX(), point.GetY()


def get_response_for_gettimeseries(get_parameters):
    """ Return json with timeseries.

    provide layers=<modelname>:<mode>, where mode is one of

    s1 (default), bath, su, vol, dep, ucx, ucy, interception, rain, evap

    options:
    quad=<quadtree index>
    absolute=true (default false): do not subtract height from s1
    messages=true/false -> for height
    maxpoints=500 -> throw away points if # > maxpoints

    csv output:
    format=csv -> defaults to 'nvd3json', option is 'csv'
    display_name=PumpXYZ  -> used as part for suggested output filename
    object_type=pumpstation -> used as part for suggested output filename
    if display_name is omitted, the name is composed of the given coordinates
    """
    # No global import, celery doesn't want this.
    from server.app import message_data

    if get_parameters.get('messages', 'false') == 'true':
        use_messages = True
    else:
        use_messages = False

    # Option to directly get the value of a quad
    quad = get_parameters.get('quad', None)
    if quad is not None:
        quad = int(quad)

    # Option for output format
    output_format = get_parameters.get('format', 'nvd3json')
    # Either a provided name (objects), or the coordinates of the clicked
    # location (2d)
    output_filename_displayname = get_parameters.get(
        'display_name', None)
    if output_filename_displayname is None:
        output_filename_displayname = ''
    if quad is not None:  # quad is either an int, or None
        output_filename_displayname = '_'.join(
            [output_filename_displayname, str(quad + 1)])  # fortran idx
    # only for csv output
    object_type = get_parameters.get('object_type', '-')

    # Fallback doesn't work: netcdf not present yet.

    # This request features a point, but an bbox is needed for reprojection.
    points = get_parameters.get('point', '10,10')
    point = np.array(map(float,
                         points.split(','))).reshape(1, 2)
    # Make a fake bounding box. Beware: units depend on epsg (wgs84)
    bbox = ','.join(map(
        str, np.array(point + np.array([[-0.0000001], [0.0000001]])).ravel()))
    get_parameters_extra = dict(height='1', width='1', bbox=bbox)
    get_parameters_extra.update(get_parameters)

    timeformat = get_parameters.get('timeformat', 'iso')  # iso or epoch
    maxpoints = get_parameters.get('maxpoints', '500')
    maxpoints = int(maxpoints)

    absolute = get_parameters.get('absolute', 'false')

    # Determine layers
    layer_parameter = get_parameters['layers']
    if ':' in layer_parameter:
        layer, mode = layer_parameter.split(':')
        get_parameters['layers'] = layer
    else:
        layer, mode = layer_parameter, 's1'

    # Get height and quad
    if use_messages:
        quad_container = message_data.get('quad_grid', **get_parameters_extra)
        dps_container = message_data.get('dps', **get_parameters_extra)

        if quad is None:
            quads, ms = get_data(container=quad_container, ma=True,
                                 **get_parameters_extra)
            quad = int(quads[0, 0])
            logger.debug('Got quads in {} ms.'.format(ms))

        dps, ms = get_data(container=dps_container, ma=True,
                           **get_parameters_extra)
        logger.debug('Got dps in {} ms.'.format(ms))

        bathymetry = -dps
        height = bathymetry[0, 0]  # not needed when absolute=true
    else:
        static_data = StaticData.get(layer=layer)
        if quad is None:
            quads, ms = get_data(container=static_data.monolith,
                                 ma=True, **get_parameters_extra)
            quad = int(quads[0, 0])
            logger.debug('Got quads in {} ms.'.format(ms))
        logger.debug('Quad = %r' % quad)

        bathymetry, ms = get_data(container=static_data.pyramid,
                                  ma=True, **get_parameters_extra)
        logger.debug('Got bathymetry in {} ms.'.format(ms))

        height = bathymetry[0, 0]

    if not height:
        logger.debug('Got no height.')
        height = 0
    logger.debug('Got height {}.'.format(height))

    # Read data from netcdf
    path = utils.get_netcdf_path(layer=get_parameters['layers'])
    if os.path.exists(path):
        with Dataset(path) as dataset:
            v = dataset.variables
            units = v['time'].getncattr('units')
            time = v['time'][:]
            # Depth values can be negative or non existent.
            # Note: all variables can be looked up here, so 'depth' is misleading.
            if mode == 's1':
                if absolute == 'false':
                    depth = np.ma.maximum(v[mode][:, quad] - height, 0).filled(0)
                else:
                    depth = v[mode][:, quad]
            else:
                if absolute == 'true':
                    # For unorm, q
                    depth = np.ma.abs(v[mode][:, quad])
                else:
                    depth = v[mode][:, quad]
            var_units = v[mode].getncattr('units')
    else:
        # dummy to prevent crashing
        logger.warning('NetCDF at [%s] does not exist (yet).' % path)
        units = 'seconds since 2015-01-01'
        time = np.array([0, 1])
        depth = np.array([0, 0])
        var_units = 'no unit'

    compressed_time = time
    compressed_depth = depth

    if compressed_time.size:
        if timeformat == 'iso':
            time_list = map(lambda t: t.isoformat(),
                            num2date(compressed_time, units=units))
        else:
            # Time in milliseconds from epoch.
            time_list = map(lambda t: 1000*float(t.strftime('%s')),
                            num2date(compressed_time, units=units))
    else:
        time_list = []
    depth_list = compressed_depth.round(3).tolist()

    # prepare header
    header = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}

    if output_format == 'nvd3json':
        while len(depth_list) > maxpoints:
            # Never throw away the last item.
            depth_list = depth_list[:-1:2] + depth_list[-1:]
            time_list = time_list[:-1:2] + time_list[-1:]
        content_dict = dict(
            timeseries=zip(time_list, depth_list),
            height=float(height),
            units=var_units)
        content = json.dumps(content_dict)
        header['content-type'] = 'application/json'
    elif output_format == 'csv':
        # 10, 10 is a virtual coordinate
        if points is not None and points != '10,10':
            # we have to create a part of the filename by using the coordinates
            coords = [float(x) for x in points.split(',')]
            # there is no good way in threedi-wms to retrieve model srs.
            # since it's only for the filename, let's look at the result rd coordinates.
            # if it is sensible, use that, else use the original coordinates.
            coords_rd = wgs_to_rd(*coords)
            if (
                coords_rd[0] > 0 and coords_rd[0] < 300000 and
                coords_rd[1] > 300000 and coords_rd[1] < 600000):
                add_displayname = ','.join(
                    [str(int(c)) for c in coords_rd])
            else:
                add_displayname = ','.join(
                    [str(c) for c in coords])
            output_filename_displayname = '_'.join([
                output_filename_displayname, add_displayname])

        # full length data
        delimiter = ','
        content_ = [CSV_HEADER]
        for datetime_, value in zip(time_list, depth_list):
            # note: quad is fortran indexing, which differs 1 from 0-based
            # netcdf indexing
            new_row = [datetime_, str(value), var_units, str(quad + 1), object_type]
            content_.append(new_row)
        content = '\n'.join([delimiter.join(r) for r in content_])
        csv_filename = 'timeseries_%s_%s' % (output_filename_displayname, mode)
        header['content-type'] = 'text/csv'  # after testing: 'test/csv'
        header['content-disposition'] = 'attachment;filename="%s.csv"' % csv_filename
    else:
        content = 'unknown format'
        header['content-type'] = 'text'

    return content, 200, header


def get_response_for_getprofile(get_parameters):
    """ Return json with profile.

    get_parameters(may be incomplete):

    use_messages: 'true' / 'false'
    line
    layers
    """

    # No global import, celery doesn't want this.
    from server.app import message_data

    if get_parameters.get('messages', 'false') == 'true':
        use_messages = True
    else:
        use_messages = False

    # Fallback doesn't work: netcdf not present yet.

    # This request features a point, but an bbox is needed for reprojection.
    # Note that GetEnvelope() returns x1, x2, y1, y2 but bbox is x1, y1, x2, y2
    geometry = ogr.CreateGeometryFromWkt(str(get_parameters['line']))
    bbox_array = np.array(
        geometry.GetEnvelope(),
    ).reshape(2, 2).transpose().ravel()
    bbox = ','.join(map(str, bbox_array))

    # set longest side to fixed size
    # Not clear why we need this...
    rastersize = 512
    x_extent = bbox_array[2] - bbox_array[0]
    y_extent = bbox_array[3] - bbox_array[1]
    aspect = (lambda x1, y1, x2, y2: (y2 - y1) / (x2 - x1))(*bbox_array)
    if aspect < 1:
        width = rastersize
        cellsize = x_extent / rastersize
        height = int(max(math.ceil(aspect * width), 1))
    else:
        height = rastersize
        cellsize = y_extent / rastersize
        width = int(max(math.ceil(height / aspect), 1))

    # Determine layer and time
    layer_parameter = get_parameters['layers']
    if ':' in layer_parameter:
        layer, mode = layer_parameter.split(':')
    else:
        layer, mode = layer_parameter, 'depth'
    time = int(get_parameters['time'])

    # Get height and quad
    get_parameters_extra = dict(width=width, height=height, bbox=bbox)
    get_parameters_extra.update(get_parameters)

    # get quads, bathymetry, depth
    if use_messages:
        time_start = _time.time()
        dps_container = message_data.get('dps', **get_parameters_extra)
        logger.debug(
            'Got containers in {} s.'.format(_time.time() - time_start))
        dps, ms = get_data(container=dps_container, ma=True,
                           **get_parameters_extra)
        logger.debug('Got dps in {} ms.'.format(ms))

        waterlevel_container = message_data.get("waterheight",
                                                **get_parameters_extra)
        logger.debug('Got waterlevel container.')
        waterlevel, ms = get_data(
            waterlevel_container, ma=True, **get_parameters_extra)

        bathymetry = -dps
        depth = waterlevel - bathymetry

        if 'sg' in message_data.grid and \
           message_data.get_raw('sg') is not None:
            # Got ground water
            quad_container = message_data.get("quad_grid",
                                              **get_parameters_extra)
            quads, ms = get_data(container=quad_container, ma=True,
                                 **get_parameters_extra)
            logger.debug('Got quads in {} ms.'.format(ms))
            groundwaterlevel = message_data.get_raw('sg')[quads]
        else:
            groundwaterlevel = np.ones(depth.shape) * np.amin(bathymetry)

        logger.debug('Got depth.')
    else:
        # Becoming obsolete
        time_start = _time.time()
        static_data = StaticData.get(layer=layer)
        quad_container = static_data.monolith
        bathy_container = static_data.pyramid
        logger.debug(
            'Got containers in {} s.'.format(_time.time() - time_start))

        quads, ms = get_data(container=quad_container, ma=True,
                             **get_parameters_extra)
        logger.debug('Got quads in {} ms.'.format(ms))

        bathymetry, ms = get_data(container=bathy_container,
                                  ma=True, **get_parameters_extra)
        logger.debug('Got bathymetry in {} ms.'.format(ms))

        # Determine the waterlevel
        dynamic_data = DynamicData.get(
            layer=layer, time=time)
        waterlevel = dynamic_data.waterlevel[quads]
        depth = waterlevel - bathymetry

        # No support for groundwater
        groundwaterlevel = np.ones(depth.shape) * np.amin(bathymetry)

    # Sample the depth using the cellsize
    magicline = vector.MagicLine(np.array(geometry.GetPoints())[:, :2])
    magicline2 = magicline.pixelize(cellsize)
    centers = magicline2.centers
    lengths_cumsum = vector.magnitude(magicline2.vectors).cumsum()
    distances = lengths_cumsum - lengths_cumsum[0]

    origin = np.array(bbox_array[[0, 3]])
    indices = tuple(np.uint64(
        np.abs(centers - origin) / cellsize,
    ).transpose())[::-1]
    depths = np.ma.maximum(depth[indices], 0)
    waterlevel_sampled = np.ma.maximum(waterlevel[indices], -100)
    groundwaterlevel_sampled = np.ma.maximum(groundwaterlevel[indices], -100)
    bathymetry_sampled = np.ma.maximum(bathymetry[indices], -100)

    # groundwaterlevel higher than the bathymetry is clipped
    groundwaterlevel_sampled = np.ma.minimum(groundwaterlevel_sampled,
                                             bathymetry_sampled)

    minimum_level = min(
        np.ma.amin(groundwaterlevel_sampled, 0),
        np.ma.amin(bathymetry_sampled, 0))
    maximum_level = max(max(
        np.ma.amax(groundwaterlevel_sampled, 0),
        np.ma.amax(bathymetry_sampled, 0)),
        np.ma.amax(waterlevel_sampled, 0))
    margin_level = max((maximum_level - minimum_level) * 0.1, 0.1)

    groundwaterlevel_delta_sampled = groundwaterlevel_sampled - minimum_level
    bathymetry_delta_sampled = bathymetry_sampled - groundwaterlevel_sampled

    compressed_depths = depths.filled(0)
    compressed_distances = distances
    compressed_bathymetry = bathymetry_delta_sampled.filled(0)
    compressed_groundwaterlevels = groundwaterlevel_delta_sampled.filled(0)

    roundfunc = lambda x: round(x, 5)
    mapped_compressed_distances = map(roundfunc, compressed_distances)

    # The bias is needed for displaying stacked graphs below zero in nv.d3.
    content = json.dumps(dict(
        depth=zip(
            mapped_compressed_distances,
            map(roundfunc, compressed_depths)),
        bathymetry_delta=zip(
            mapped_compressed_distances,
            map(roundfunc, compressed_bathymetry)),
        groundwater_delta=zip(
            mapped_compressed_distances,
            map(roundfunc, compressed_groundwaterlevels)),
        offset=zip(
            mapped_compressed_distances,
            [roundfunc(minimum_level)]*len(mapped_compressed_distances)),
        summary=dict(minimum=minimum_level, maximum=maximum_level,
                     margin=margin_level),
    ))

    return content, 200, {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getquantity(get_parameters):
    """ Return json with quantity for all calculation cells.

    Option to return pumps, weirs and orifices: requires messages.
    """
    # Determine layer and time
    layer = get_parameters['layers']
    time = int(get_parameters.get('time', 0))
    quantity = get_parameters['quantity']
    quantities = quantity.split(',')

    # Do we need message data? Intersect quantities and messages vars
    if set(quantities) & set(PANDAS_VARS):
        # No global import, celery doesn't want this.
        from server.app import message_data

    try:
        decimals = int(get_parameters['decimals'])
    except KeyError:
        decimals = None

    # get the flow link numbers from redis; N.B. link numbers are returned
    # as strings from redis
    model_slug = utils.get_loaded_model()
    link_numbers = rc_node.smembers('models:%s:link_numbers' % model_slug)

    # Load quantity from netcdf
    netcdf_path = utils.get_netcdf_path(layer)
    with Dataset(netcdf_path) as dataset:
        # Explicitly make a masked array. Some quantities (unorm, q) return an
        # ndarray.
        data = {}
        for quantity_key in quantities:
            # Special pandas variables
            if quantity_key in PANDAS_VARS:
                # TODO: link_number trick, but for objects.
                data[quantity_key] = message_data.get_pandas(quantity_key)
                continue
            if quantity_key not in KNOWN_VARS:
                continue  # skip unknown variables
            # Normal variables
            if link_numbers:
                # to convert to np.uint64, link_numbers need to be converted to
                # a list first
                np_link_numbers = np.uint64(list(link_numbers))
                ma = np.ma.masked_array(
                    dataset.variables[quantity_key][time][np_link_numbers])
                if decimals is None:
                    quantity_data = dict(zip(link_numbers, ma.filled().
                                             tolist()))
                else:
                    quantity_data = dict(
                        zip(link_numbers, ma.filled().round(decimals).
                            tolist()))
            else:
                # no flow link numbers for filtering, so return all
                # this produces the original unfiltered data dict
                ma = np.ma.masked_array(dataset.variables[quantity_key][time])
                if decimals is None:
                    quantity_data = dict(enumerate(ma.filled().tolist()))
                else:
                    quantity_data = dict(enumerate(ma.filled().round(
                        decimals).tolist()))
            data[quantity_key] = quantity_data

    content = json.dumps(dict(nodatavalue=ma.fill_value, data=data))

    return content, 200, {'content-type': 'application/json',
                          'Access-Control-Allow-Origin': '*',
                          'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getcontours(get_parameters):
    """ Return json with quantity for all calculation cells. """

    # Determine layer and time
    layer = get_parameters['layers']

    # Load contours from netcdf
    netcdf_path = utils.get_netcdf_path(layer)
    with Dataset(netcdf_path) as dataset:
        x = dataset.variables['FlowElemContour_x'][:]
        y = dataset.variables['FlowElemContour_y'][:]
    contours = dict(enumerate(np.dstack([x, y]).tolist()))
    content = json.dumps(dict(contours=contours))
    return content, 200, {'content-type': 'application/json',
                          'Access-Control-Allow-Origin': '*',
                          'Access-Control-Allow-Methods': 'GET'}


class StaticData(object):
    """
    Container for static data from the netcdf.
    """
    @classmethod
    def get(cls, layer, reload=False):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        if reload:
            value = cls(layer=layer, reload=reload)
            return value

        value = cls(layer=layer)
        return value

    def __init__(self, layer, reload=False):
        """Init pyramid and monolith, and order creation if necessary."""
        logger.debug('Initializing StaticData for {}'.format(layer))
        errors = []
        # Initialize pyramid for bathymetry
        pyramid_path = utils.get_pyramid_path(layer)
        pyramid = rasters.Pyramid(path=pyramid_path, compression='DEFLATE')

        # Order building if necessary
        if not pyramid.has_data():
            tasks.make_pyramid.delay(layer)
            errors.append('Pyramid not ready yet, task submitted.')
        # If all ok, set pyramid attribute.
        self.pyramid = pyramid

        # Initialize monolith for quad layout
        monolith_path = os.path.join(config.CACHE_DIR, layer, 'monolith')
        monolith = rasters.Monolith(path=monolith_path, compression='DEFLATE')

        # Order building if necessary
        # TODO: this can be initiated multiple times, that's unnecessary
        if not monolith.has_data():
            tasks.make_monolith.delay(layer=layer)
            errors.append('Monolith not ready yet, task submitted.')

        if errors:
            raise ValueError(' '.join(errors))

        # If all ok, set monolith attribute.
        self.monolith = monolith


class DynamicData(object):
    """
    Container for only the waterlevel data from the netcdf.
    """
    @classmethod
    def get(cls, layer, time, variable='s1', netcdf_path=None):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        value = cls(layer=layer, time=time, variable=variable,
                    netcdf_path=netcdf_path)
        return value

    def __init__(self, layer, time, variable='s1', netcdf_path=None):
        """ Load data from netcdf. """
        if netcdf_path is None:
            netcdf_path = utils.get_netcdf_path(layer)
        with Dataset(netcdf_path) as dataset:
            waterlevel_variable = dataset.variables[variable]

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
