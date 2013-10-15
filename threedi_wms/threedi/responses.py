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
from scipy import ndimage
from matplotlib import cm
from matplotlib import colors

import numpy as np
import ogr

import collections
import datetime
import io
import json
import logging
import math
import os
import shutil


cache = {}
ogr.UseExceptions()


def rgba2image(rgba, antialias=1):
    """ return imagedata. """
    size = [d // antialias for d in rgba.shape[1::-1]]
    image = Image.fromarray(rgba).resize(size, Image.ANTIALIAS)

    buf = io.BytesIO()
    image.save(buf, 'png')
    return buf.getvalue()


def get_depth_image(masked_array, waves=None, antialias=1, hmin=0, hmax=2):
    """ Return a png image from masked_array. """
    # Hardcode depth limits, until better height data
    normalize = colors.Normalize(vmin=hmin, vmax=hmax)
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
    arr = masked_array
    #import pdb; pdb.set_trace()
    if waves is not None:
        arr += waves
    rgba = colormap(normalize(arr), bytes=True)
    # Make negative depths transparent
    rgba[..., 3][np.ma.less_equal(masked_array, 0)] = 0

    return rgba2image(rgba=rgba, antialias=antialias)


def get_bathymetry_image(masked_array, limits, antialias=1):
    """ Return imagedata. """
    normalize = colors.Normalize(vmin=limits[0], vmax=limits[1])
    colormap = cm.summer
    rgba = colormap(normalize(masked_array), bytes=True)
    return rgba2image(rgba=rgba, antialias=antialias)


def get_grid_image(masked_array, antialias=1):
    """ Return imagedata. """
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
    return rgba2image(rgba=rgba, antialias=antialias)


def get_water_waves(masked_array, anim_frame, antialias=1):
    """
    Calculate waves from velocity array
    """
    # Animating 'waves'
    y_shape, x_shape = masked_array.shape
    x, y = np.mgrid[0:y_shape, 0:x_shape]
    offset = anim_frame * 0.01
    period = masked_array.filled(1)
    amplitude = masked_array.filled(0)
    waves = (np.sin(np.pi * 64 / period *
             (offset + x / x_shape + y / y_shape)) * amplitude +
             np.sin(np.pi * 60 / period *
             (offset + y / y_shape)) * amplitude)

    # 'Shade' by convolution
    waves_shade = ndimage.filters.convolve(
        waves,
        np.array([[-.2, -0.5, -0.7, -.5, .3],
                  [-.5, -0.7, -1.5,  .4, .5],
                  [-.7, -1.5,  0.0, 1.5, .7],
                  [-.5, -0.4,  1.5,  .7, .5],
                  [-.3,  0.5,  0.7,  .5, .2]]))

    normalize = colors.Normalize(vmin=0, vmax=24)

    return get_depth_image(masked_array,
                           antialias=antialias,
                           waves=normalize(waves_shade))


def get_data(container, ma=False, **get_parameters):
    """
    Return numpy (masked) array from container
    """
    start = datetime.datetime.now()
    # Derive properties from get_paramaters
    if get_parameters.get('antialias', 'no') == 'yes':
        antialias = 2
    else:
        antialias = 1
    size = (antialias * int(get_parameters['width']),
            antialias * int(get_parameters['height']))
    extent = map(float, get_parameters['bbox'].split(','))
    srs = get_parameters['srs']

    # Create dataset
    geometry = rasters.DatasetGeometry(size=size, extent=extent)
    dataset = geometry.to_dataset(
        datatype=container.datatype,
        projection=srs,
    )
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


# Responses for various requests
def get_response_for_getmap(get_parameters):
    """ Return png image. """
    # Get the quad and waterlevel data objects
    layer_parameter = get_parameters['layers']
    if ':' in layer_parameter:
        layer, mode = layer_parameter.split(':')
    else:
        layer, mode = layer_parameter, 'depth'

    rebuild_static = get_parameters.get('rebuild_static', 'no') == 'yes'
    if rebuild_static:
        logging.debug('Got rebuild_static {}, deleting cache.'.format(layer))
        # delete var/cache/3di/<model> directory
        # make sure layer has no directories or whatsoever.
        cache_path = os.path.join(config.CACHE_DIR, layer.replace('/', ''))
        shutil.rmtree(cache_path)

    try:
        static_data = StaticData.get(layer=layer, reload=rebuild_static)
    except ValueError:
        return 'Objects not ready, starting preparation.'
    except rasters.LockError:
        return 'Objects not ready, preparation in progress.'

    if mode in ['depth', 'bathymetry', 'flood']:
        bathymetry, ms = get_data(container=static_data.pyramid,
                                  ma=True, **get_parameters)
        logging.debug('Got bathymetry in {} ms.'.format(ms))
    if mode in ['depth', 'grid', 'flood']:
        quads, ms = get_data(container=static_data.monolith,
                             ma=True, **get_parameters)
        logging.debug('Got quads in {} ms.'.format(ms))

    if get_parameters.get('antialias', 'no') == 'yes':
        antialias = 2
    else:
        antialias = 1

    if get_parameters.get('nocache', 'no') == 'yes':
        use_cache = False
    else:
        use_cache = True

    if mode == 'depth':
        hmax = get_parameters.get('hmax', 2.0)
        time = int(get_parameters['time'])
        dynamic_data = DynamicData.get(
            layer=layer, time=time, use_cache=use_cache)
        waterlevel = dynamic_data.waterlevel[quads]
        depth = waterlevel - bathymetry

        if 'anim_frame' in get_parameters:
            # Add wave animation
            content = get_water_waves(
                masked_array=depth,
                anim_frame=int(get_parameters['anim_frame']),
                antialias=antialias,
                hmax=hmax
            )
        else:
            # Direct image
            content = get_depth_image(masked_array=depth,
                                      antialias=antialias,
                                      hmax=hmax)
    elif mode == 'flood':
        # time is actually the sequence number of the flood
        hmax = get_parameters.get('hmax', 2.0)
        time = int(get_parameters['time'])
        dynamic_data = DynamicData.get(
            layer=layer, time=time, use_cache=use_cache, variable='floodfill',
            netcdf_path=utils.get_netcdf_path_flood(layer)
        )
        waterlevel = dynamic_data.waterlevel[quads]
        depth = waterlevel - bathymetry

        if 'anim_frame' in get_parameters:
            # Add wave animation
            content = get_water_waves(
                masked_array=depth,
                anim_frame=int(get_parameters['anim_frame']),
                antialias=antialias
            )
        else:
            # Direct image
            content = get_depth_image(masked_array=depth,
                                      antialias=antialias,
                                      hmax=hmax)
    elif mode == 'bathymetry':
        limits = map(float, get_parameters['limits'].split(','))
        content = get_bathymetry_image(masked_array=bathymetry,
                                       limits=limits,
                                       antialias=antialias)
    elif mode == 'grid':
        content = get_grid_image(masked_array=quads,
                                 antialias=antialias)

    return content, 200, {
        'content-type': 'image/png',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getinfo(get_parameters):
    """ Return json with bounds and timesteps. """
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
        bathy_path = utils.get_bathymetry_path(layer=get_parameters['layers'])
        # It defaults to Rijksdriehoek RD
        source_projection = utils.get_bathymetry_srs(bathy_path)

        logging.info('Source projection: %r' % source_projection)
        #source_projection = 22234 if 'kaapstad' in path.lower() else rasters.RD
        target_projection = srs
        extent = gislib_utils.get_transformed_extent(
            extent=netcdf_extent,
            source_projection=source_projection,
            target_projection=target_projection,
        )
    else:
        logging.warning('No srs data available.')
        extent = netcdf_extent

    # Prepare response
    content = json.dumps(dict(bounds=extent,
                              limits=limits,
                              timesteps=timesteps))
    return content, 200, {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_gettimeseries(get_parameters):
    """ Return json with timeseries.

    provide layers=<modelname>:<mode>, where mode is one of

    s1 (default), bath, su, vol, dep, ucx, ucy, interception, rain, evap 
    """
    # This request features a point, but an bbox is needed for reprojection.
    point = np.array(map(float,
                         get_parameters['point'].split(','))).reshape(1, 2)
    #bbox = ','.join(map(str, np.array(point + np.array([[-1], [1]])).ravel()))
    # Make a fake bounding box. Beware: units depend on epsg (wgs84)
    bbox = ','.join(map(str, np.array(point + np.array([[-0.0000001], [0.0000001]])).ravel()))
    get_parameters_extra = dict(height='1', width='1', bbox=bbox)
    get_parameters_extra.update(get_parameters)
    quad = get_parameters.get('quad', None)
    if quad is not None:
        quad = int(quad)

    # Determine layers
    layer_parameter = get_parameters['layers']
    if ':' in layer_parameter:
        layer, mode = layer_parameter.split(':')
        get_parameters['layers'] = layer
    else:
        layer, mode = layer_parameter, 's1'

    # Get height and quad
    static_data = StaticData.get(layer=layer)
    quads, ms = get_data(container=static_data.monolith,
                         ma=True, **get_parameters_extra)
    if quad is None:
        quad = int(quads[0, 0])
    logging.debug('Quad = %r' % quad)
    logging.debug('Got quads in {} ms.'.format(ms))

    bathymetry, ms = get_data(container=static_data.pyramid,
                              ma=True, **get_parameters_extra)
    height = bathymetry[0, 0]
    logging.debug('Got bathymetry in {} ms.'.format(ms))

    # Read data from netcdf
    path = utils.get_netcdf_path(layer=get_parameters['layers'])
    with Dataset(path) as dataset:
        v = dataset.variables
        units = v['time'].getncattr('units')
        time = v['time'][:]
        # Depth values can be negative or non existent.
        if mode == 's1':
            depth = np.ma.maximum(v[mode][:, quad] - height, 0).filled(0)
        else:
            depth = np.ma.maximum(v[mode][:, quad], 0).filled(0)
        var_units = v[mode].getncattr('units')

    compressed_time = time
    compressed_depth = depth

    if compressed_time.size:
        time_list = map(lambda t: t.isoformat(),
                        num2date(compressed_time, units=units))
    else:
        time_list = []
    depth_list = compressed_depth.round(3).tolist()

    content_dict = dict(
        timeseries=zip(time_list, depth_list), 
        height=float(height),
        units=var_units)
    content = json.dumps(content_dict)
    return content, 200, {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getprofile(get_parameters):
    """ Return json with profile. """
    # This request features a point, but an bbox is needed for reprojection.
    # Note that GetEnvelope() returns x1, x2, y1, y2 but bbox is x1, y1, x2, y2
    geometry = ogr.CreateGeometryFromWkt(str(get_parameters['line']))
    bbox_array = np.array(
        geometry.GetEnvelope(),
    ).reshape(2, 2).transpose().ravel()
    bbox = ','.join(map(str, bbox_array))

    # set longest side to fixed size
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
    static_data = StaticData.get(layer=layer)
    quads, ms = get_data(container=static_data.monolith,
                         ma=True, **get_parameters_extra)
    logging.debug('Got quads in {} ms.'.format(ms))

    bathymetry, ms = get_data(container=static_data.pyramid,
                              ma=True, **get_parameters_extra)
    logging.debug('Got bathymetry in {} ms.'.format(ms))

    # Determine the waterlevel
    dynamic_data = DynamicData.get(
        layer=layer, time=time, use_cache=False)
    waterlevel = dynamic_data.waterlevel[quads]
    depth = waterlevel - bathymetry

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
    bathymetry_sampled = np.ma.maximum(bathymetry[indices], -100)

    #bathymetry from 0 up
    bathymetry_minimum = min(np.ma.amin(bathymetry_sampled, 0), 0)
    bathymetry_sampled = bathymetry_sampled - bathymetry_minimum

    compressed_depths = depths.filled(0)
    compressed_distances = distances
    compressed_waterlevels = waterlevel_sampled.filled(0)
    compressed_bathymetry = bathymetry_sampled.filled(0)

    roundfunc = lambda x: round(x, 5)
    mapped_compressed_distances = map(roundfunc, compressed_distances)

    # The bias is needed for displaying stacked graphs below zero in nv.d3.
    content = json.dumps(dict(
        depth=zip(
            mapped_compressed_distances,
            map(roundfunc, compressed_depths)),
        # waterlevel=zip(
        #     mapped_compressed_distances,
        #     map(roundfunc, compressed_waterlevels)),
        bathymetry=zip(
            mapped_compressed_distances,
            map(roundfunc, compressed_bathymetry)),
        bias=zip(mapped_compressed_distances,
            [roundfunc(bathymetry_minimum)]*len(mapped_compressed_distances)),
    ))

    return content, 200, {
        'content-type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'}


def get_response_for_getquantity(get_parameters):
    """ Return json with quantity for all calculation cells. """

    # Determine layer and time
    layer = get_parameters['layers']
    time = int(get_parameters['time'])
    quantity = get_parameters['quantity']
    try:
        decimals = int(get_parameters['decimals'])
    except KeyError:
        decimals = None

    # Load quantity from netcdf
    netcdf_path = utils.get_netcdf_path(layer)
    with Dataset(netcdf_path) as dataset:
        ma = dataset.variables[quantity][time]
    nodatavalue = ma.fill_value
    if decimals is None:
        data = dict(enumerate(ma.filled().tolist()))
    else:
        data = dict(enumerate(ma.filled().round(decimals).tolist()))
    content = json.dumps(dict(nodatavalue=nodatavalue, data=data))
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
        # Prepare key
        key = collections.namedtuple(
            'StaticDataKey', ['layer'],
        )(layer=layer)

        if reload:
            value = cls(layer=layer, reload=reload)
            cache[key] = value
            return value

        # Return object
        try:
            return cache[key]
        except KeyError:
            value = cls(layer=layer)
            cache[key] = value
            return value

    def __init__(self, layer, reload=False):
        """ Init pyramid and monolith, and order creation if necessary. """
        logging.debug('Initializing StaticData for {}'.format(layer))
        errors = []
        # Initialize pyramid for bathymetry
        pyramid_path = utils.get_pyramid_path(layer)
        pyramid = rasters.Pyramid(path=pyramid_path,
                                 compression='DEFLATE')

        # Order building if necessary
        if not pyramid.has_data():
            tasks.make_pyramid.delay(layer)
            errors.append('Pyramid not ready yet, task submitted.')
            #raise ValueError('Pyramid not ready yet, task submitted.')
        # If all ok, set pyramid attribute.
        self.pyramid = pyramid

        # Initialize monolith for quad layout
        monolith_path = os.path.join(config.CACHE_DIR, layer, 'monolith')
        monolith = rasters.Monolith(path=monolith_path, compression='DEFLATE')

        # Order building if necessary
        # TODO: this can be initiated multiple times, that's unnecessary
        if not monolith.has_data():
            tasks.make_monolith.delay(layer=layer)
            errors.append('Pyramid not ready yet, task submitted.')
            #raise ValueError('Monolith not ready yet, task submitted.')

        if errors:
            raise ValueError(' '.join(errors))

        # If all ok, set monolith attribute.
        self.monolith = monolith


class DynamicData(object):
    """
    Container for only the waterlevel data from the netcdf.
    """
    @classmethod
    def get(cls, layer, time, use_cache, variable='s1', netcdf_path=None):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        # Prepare key
        key = collections.namedtuple(
            'DynamicDataKey', ['layer', 'time', 'variable', 'netcdf_path'],
        )(layer=layer, time=time, variable=variable, netcdf_path=netcdf_path)
        # Return object
        if use_cache:
            try:
                return cache[key]
            except KeyError:
                value = cls(layer=layer, time=time, variable=variable,
                            netcdf_path=netcdf_path)
                cache[key] = value
                return value
        else:
            value = cls(layer=layer, time=time, variable=variable,
                        netcdf_path=netcdf_path)
            cache[key] = value
            return value

    def __init__(self, layer, time, variable='s1', netcdf_path=None):
        """ Load data from netcdf. """
        #logging.debug('Loading dynamic data layer {}...'.format(layer))
        if netcdf_path is None:
            netcdf_path = utils.get_netcdf_path(layer)
        with Dataset(netcdf_path) as dataset:
            waterlevel_variable = dataset.variables[variable]
            #logging.debug(waterlevel_variable)

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
