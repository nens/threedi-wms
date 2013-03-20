# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from threedi import config
from gislib import raster

from PIL import Image
from netCDF4 import Dataset
from scipy import spatial
from scipy import ndimage
from matplotlib import colors
from osgeo import gdal

import numpy as np
import requests

import collections
import datetime
import io
import json
import logging
import os

subgrid_cache = {}


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


def get_quad_dataset(static_data):
    """ Get a gdaldataset containing the quad positions. """
    # Determine the grid based on the smallest quads.
    x1, y1, x2, y2 = static_data.extent
    width = int(round((x2 - x1) / static_data.widths.min()))
    height = int(round((y2 - y1) / static_data.heights.min()))
    geometry = raster.DatasetGeometry(
        extent=static_data.extent, size=(width, height),
    )
    gridpoints = geometry.gridpoints()
    quad_grid = np.ma.array(np.empty((height, width)), mask=True)

    for area in np.unique(static_data.areas):

        index = (static_data.areas == area)

        # Prepare an array with indices to current quads
        count = index.sum()
        quad_index = np.ma.array(np.empty(count + 1), mask=True)
        quad_index[:count] = np.arange(index.size)[index]

        # Construct and query a nearest-neighbour interpolator
        upper = np.sqrt(area) / 2
        logging.debug('Adding quads of area {} to dataset.'.format(area))
        data_index = spatial.cKDTree(static_data.centers[index]).query(
            gridpoints, p=np.inf, distance_upper_bound=upper,
        )[1]

        # Add to result
        quad_grid = np.ma.sum([
            quad_grid,
            quad_index[data_index].reshape(height, width)
        ], axis=0)

    dataset = geometry.to_dataset(datatype=gdal.GDT_UInt32)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(static_data.areas.size)
    band.WriteArray(quad_grid.filled(-9999.))
    return dataset


def get_height_from_server(geometry):
    """ On the server an rgba encoder height map is prepared. """
    extent, width, height = (geometry.extent,
                             geometry.width,
                             geometry.height)

    # Get rgba image from geoserver
    buf = io.BytesIO()
    buf.write(requests.get(
        config.GEOSERVER_URL,
        params=dict(
            LAYERS=config.GEOSERVER_LAYER,
            STYLES='',
            FORMAT='image/png',
            SERVICE='WMS',
            VERSION='1.1.1',
            REQUEST='GetMap',
            SRS='EPSG:28992',
            BBOX=','.join(map(str, extent)),
            WIDTH=str(width),
            HEIGHT=str(height),
        )
    ).content)
    buf.seek(0)
    image = Image.open(buf)

    # Convert to masked array of floats
    data = np.fromstring(
        np.array(image).tostring(),
        dtype=np.float32,
    ).reshape(height, width)
    mask = np.isnan(data)
    result = np.ma.array(data, mask=mask)

    return result


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
    waves = (np.sin(np.pi*64/magnitude*(offset+x/x_shape+y/y_shape))*magnitude +
             np.sin(np.pi*60/magnitude*(offset+y/y_shape))*magnitude)

    # 'Shade' by convolution
    waves_shade = ndimage.filters.convolve(
        waves, 
        np.array([[-.2, -.5, -.7,-.5, .3],
                  [-.5, -.7,-1.5, .4, .5],
                  [-.7,-1.5,   0,1.5, .7],
                  [-.5, -.4, 1.5, .7, .5],
                  [-.3,  .5,  .7, .5, .2]]))

    normalize = colors.Normalize(vmin=0, vmax=24)

    return get_image(masked_array, waves=normalize(waves_shade))


def get_height_from_file(data, get_parameters):
    """ Return numpy masked array. """
    projection = get_parameters['srs']
    geometry = raster.DatasetGeometry(
        extent=map(float, get_parameters['bbox'].split(',')),
        size=(int(get_parameters['width']),
              int(get_parameters['height'])),
    )

    # Get height from file
    path = data.path
    bathimetry = gdal.Open(path.replace('.nc', '.tif'))
    height_datatype = bathimetry.GetRasterBand(1).DataType
    height_nodatavalue = bathimetry.GetRasterBand(1).GetNoDataValue()
    ds_height = geometry.to_dataset(datatype=height_datatype)
    ds_height.SetProjection(projection)
    raster.reproject(source=bathimetry,
                    target=ds_height,
                    algorithm=gdal.GRA_NearestNeighbour)
    height_data = ds_height.ReadAsArray()
    height = np.ma.array(height_data,
                         mask=np.equal(height_data, height_nodatavalue))

    return height


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
    path = os.path.join(config.DATA_DIR,
                        get_parameters['dataset'])
    static_data = StaticData(path=path)

    # Determine extent
    srs = get_parameters['srs']
    if srs:
        source_projection = raster.RD
        target_projection = srs
        extent = raster.get_transformed_extent(
            extent=static_data.extent,
            source_projection=source_projection,
            target_projection=target_projection,
        )
    else:
        extent = static_data.extent

    # Prepare response
    content = json.dumps(dict(bounds=extent,
                              timesteps=static_data.timesteps))
    return content, 200, {'content-type': 'application/json'}


def get_response_for_prepare(get_parameters):
    """
    Prepare geo objects.
    """
    geo_object_names = dict(height_monolith='Height Monolith',
                            height_pyramid='Height Pyramid',
                            quad_monolith='Quad Monolith',
                            quad_pyramid='Quad Pyramid')

    prepare_dataset = get_parameters['dataset']
    prepare_type = get_parameters['type']
    path = os.path.join(config.DATA_DIR,
                        prepare_dataset)
    static_data = StaticData.get(path=path)

    geo_object = getattr(static_data, prepare_type)
    if not geo_object.has_data():
        if prepare_type.startswith('height'):
            geo_object.add(static_data.height_container.dataset)
        if prepare_type.startswith('quad'):
            geo_object.add(static_data.quad_container.dataset)
    return '{} O.K. for {}.'.format(
        geo_object_names[prepare_type], prepare_dataset,
    )


def get_response_for_getmap(get_parameters):
    """ Return png image. """
    # Get the quad and waterlevel data objects
    path = os.path.join(config.DATA_DIR,
                        get_parameters['dataset'])
    time = int(get_parameters['time'])
    anim_frame = get_parameters.get('anim_frame', None)
    if anim_frame is not None:
        anim_frame = int(anim_frame)

    static_data = StaticData.get(path=path)
    dynamic_data = DynamicData.get(path=path, time=time)

    # Get height
    height_time = datetime.datetime.now()
    if static_data.height_pyramid.has_data():
        height = get_array(container=static_data.height_pyramid,
                           ma=True,
                           **get_parameters)
        height_source = 'pyramid'
    elif static_data.height_monolith.has_data():
        height = get_array(container=static_data.height_monolith,
                           ma=True,
                           **get_parameters)
        height_source = 'monolith'
    else:
        height = get_array(container=static_data.height_container,
                           ma=True,
                           **get_parameters)
        height_source = 'container ({})'.format(os.path.basename(
            static_data.height_container.dataset.GetFileList()[0],
        ))
    logging.debug('Got height from {} in {} ms.'.format(
        height_source,
        1000 * (datetime.datetime.now() - height_time).total_seconds(),
    ))
    # Or height from ahn geoserver server?

    # Get waterlevel
    quad_time = datetime.datetime.now()
    if static_data.quad_monolith.has_data():
        quad = get_array(container=static_data.quad_monolith,
                         ma=False,
                         **get_parameters)
        quad_source = 'monolith'
    elif static_data.quad_pyramid.has_data():
        quad = get_array(container=static_data.quad_pyramid,
                         ma=False,
                         **get_parameters)
        quad_source = 'pyramid'
    else:
        quad = get_array(container=static_data.quad_container,
                         ma=False,
                         **get_parameters)
        quad_source = 'container'
    logging.debug('Got quad from {} in {} ms.'.format(
        quad_source,
        1000 * (datetime.datetime.now() - quad_time).total_seconds(),
    ))

    waterlevel = dynamic_data.waterlevel[quad]

    # Combine and return response
    if anim_frame is None:
        content = get_image(waterlevel - height)
    else:
        content = get_water_waves(waterlevel - height, anim_frame)
    return content, 200, {'content-type': 'image/png'}


class StaticData(object):
    """
    Container for static data from the netcdf.
    """
    @classmethod
    def get(cls, path):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        # Prepare key
        key = collections.namedtuple(
            'SubgridQuadKey', ['path'],
        )(path=path)

        # Return object
        try:
            return subgrid_cache[key]
        except KeyError:
            value = cls(path)
            subgrid_cache[key] = value
            return value

    def __init__(self, path):
        """ Load data from netcdf. """
        # Load data
        with Dataset(path) as dataset:
            v = dataset.variables
            fex, fey = v['FlowElemContour_x'][:], v['FlowElemContour_y'][:]
            fcx, fcy = v['FlowElem_xcc'][:], v['FlowElem_ycc'][:]
            self.timesteps = v['s1'].shape[0]
        x1, x2 = fex.min(1), fex.max(1)
        y1, y2 = fey.min(1), fey.max(1)

        # Set convenient attributes
        self.widths = x2 - x1
        self.heights = y2 - y1
        self.areas = self.widths * self.heights
        self.centers = np.array([fcx, fcy]).T
        self.extent = (x1.min(), y1.min(),
                       x2.max(), y2.max())

        # Prepare to initialize the geo-objects
        name = os.path.basename(path)
        root, ext = os.path.splitext(name)

        # Prepare height geo-objects
        self.height_pyramid = raster.Pyramid(
            os.path.join(config.CACHE_DIR, root, 'height_pyramid'),
            compression='DEFLATE',
        )
        self.height_monolith = raster.Monolith(
            os.path.join(config.CACHE_DIR, root, 'height_monolith'),
            compression='DEFLATE',
            memory=False,
        )
        # Allow for fallback to asc or zipped asc, but really, don't.
        files = os.listdir(config.DATA_DIR)
        if root + '.tif' in files:
            path = os.path.join(config.DATA_DIR, root + '.tif')
        elif root + '.zip' in files:
            path = os.path.join('/vsizip//',
                                config.DATA_DIR.lstrip('/'),
                                root + '.zip')
        elif root + '.asc' in files:
            path = os.path.join(config.DATA_DIR, root + '.asc')
        logging.debug(path)
        self.height_container = raster.Container(path)

        # Prepare quad geo-objects
        self.quad_pyramid = raster.Pyramid(
            os.path.join(config.CACHE_DIR, root, 'quad_pyramid'),
            compression='DEFLATE',
        )
        if self.height_pyramid.has_data():
            self.quad_pyramid.cellsize = self.height_pyramid.cellsize
        self.quad_monolith = raster.Monolith(
            os.path.join(config.CACHE_DIR, root, 'quad_monolith'),
            compression='DEFLATE',
            memory=True,
        )
        if self.quad_monolith.has_data():
            quad_dataset = self.quad_monolith.dataset  # Much faster
        else:
            quad_dataset = get_quad_dataset(self)
        self.quad_container = raster.Container(dataset=quad_dataset)


class DynamicData(object):
    """
    Container for only the waterlevel data from the netcdf.
    """
    @classmethod
    def get(cls, path, time):
        """
        Return instance from cache if possible, new instance otherwise.
        """
        # Prepare key
        key = collections.namedtuple(
            'SubgridWaterlevelKey', ['path', 'time'],
        )(path=path, time=time)

        # Return object
        try:
            return subgrid_cache[key]
        except KeyError:
            value = cls(path=path, time=time)
            subgrid_cache[key] = value
            return value

    def __init__(self, path, time):
        """ Load data from netcdf. """
        with Dataset(path) as dataset:
            waterlevel_variable = dataset.variables['s1']

            # Initialize empty array with one element more than amount of quads
            self.waterlevel = np.ma.array(
                np.empty(waterlevel_variable.shape[1] + 1),
                mask=True,
            )

            # Fill with waterlevel from netcdf
            self.waterlevel[0:-1] = waterlevel_variable[time]
