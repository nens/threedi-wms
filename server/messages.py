from mmi import recv_array
from gislib import rasters

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np
from netCDF4 import Dataset

import osgeo.gdal
import osgeo.osr
import bisect
import sys
import traceback
import os
import json

from server import config

from threading import BoundedSemaphore
from math import trunc

from . import dump_data

DELTARES_NO_DATA = 1e10
NO_DATA = -9999

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# global zmq context
ctx = zmq.Context()


UPDATE_INDICES_VARS = [
    'nod_type', 'imaxk', 'nodk', 'jmaxk', 'nodm', 'nodn',
    'dxp', 'x0p', 'dyp', 'y0p', 'x1p', 'y1p', 'imax', 'jmax', 'wkt']
DEPTH_VARS = [
    'dps', 'quad_grid']


class Listener(threading.Thread):
    def __init__(self, socket, message_data, *args, **kwargs):
        self.socket = socket
        self.message_data = message_data
        threading.Thread.__init__(self, *args, **kwargs)
        # A flag to notify the thread that it should finish up and exit
        self.kill_received = False

    def reset_grid_data(self):
        logger.debug('Resetting grid data...')
        for k in self.message_data.grid.keys():
            del self.message_data.grid[k]  # try to save memory
        self.message_data.interpolation_ready = False
        self.message_data.pandas = {}

    def _run(self):
        """run the thread"""
        message_data = self.message_data
        socket = self.socket
        while not self.kill_received:
            arr, metadata = recv_array(socket)

            if metadata['action'] == 'reset':
                self.reset_grid_data()
            elif metadata['action'] == 'update':
                logger.debug('Updating grid data [%s]' % metadata['name'])
                if 'model' in metadata:
                    restarted = (
                        metadata['name'] == 't1' and
                        metadata['sim_time_seconds'] < 0.1)
                    if (metadata['model'] != message_data.loaded_model
                            or restarted):
                        # New model detected
                        logger.info('New model detected: %r (old=%r)' % (
                            metadata['model'], message_data.loaded_model))

                        # Double reset algorithm.
                        message_data.grid = {}
                        message_data.interpolation_ready = False
                        message_data.loaded_model = metadata['model']

                # Update new grid
                if metadata['name'] in message_data.grid:
                    del message_data.grid[metadata['name']]  # saves memory
                if arr.dtype.kind == 'S':
                    # String, for wkt
                    message_data.grid[metadata['name']] = ''.join(arr)
                else:
                    message_data.grid[metadata['name']] = arr

                if (all(v in message_data.grid for v in DEPTH_VARS) and
                        metadata['name'] in DEPTH_VARS):

                    logger.debug(
                        'Update grids after receiving dps or quad_grid...')
                    message_data.update_grids()
                    logger.debug('Update grids finished.')

                # check update indices
                if (all(v in message_data.grid for v in UPDATE_INDICES_VARS)
                        and metadata['name'] in UPDATE_INDICES_VARS):

                    logger.debug('Update indices...')
                    message_data.X, message_data.Y, message_data.L = (
                        message_data.calc_indices())
                    logger.debug('Update indices finished.')
            elif metadata['action'] == 'update-pandas':
                logger.debug('Update pandas data [%s]...', metadata['name'])
                # TODO: in case of weir, delete unused variables.
                message_data.pandas[metadata['name']] = json.loads(
                    metadata['pandas_json'])

            elif metadata['action'] == 'dump':
                dump_data.dump_data(
                    metadata['output_filename'], metadata['input_path'],
                    message_data)
            else:
                logger.debug('Got an unknown message: %r' % metadata)

    def run(self):
        """Run the thread fail-safe"""
        while not self.kill_received:
            try:
                self._run()
            except:
                logger.error('An unknown severe error occured.')
                # Log everything
                exc_type, exc_value, exc_traceback = sys.exc_info()
                for line in traceback.format_exception(
                        exc_type, exc_value, exc_traceback):
                    logger.error(line)

                # Throw away existing data: can be corrupt.
                # you have to restart a model or 'beam to wms'.
                self.reset_grid_data()


class MessageData(object):
    """
    Container for model message data
    """
    def make_listener(self, sub_port):
        """make a socket that waits for new data in a thread"""
        subsock = ctx.socket(zmq.SUB)
        subsock.connect("tcp://localhost:{port}".format(port=sub_port))
        subsock.setsockopt(zmq.SUBSCRIBE, b'')
        thread = Listener(subsock, self)
        thread.daemon = True
        thread.start()
        self.thread = thread
        # In a hook of the website: thread.kill_received = True

    def stop_listener(self):
        if self.thread and self.thread.isAlive:
            logger.debug("Killing listener in thread {}".format(self.thread))
            self.thread.kill_received = True

    def calc_indices(self, grid=None):
        """
        create all the indices that we need for performance

        These vars probably use a lot of memory.
        """
        # lookup cell centers
        if grid is None:
            grid = self.grid

        # twod_idx is a boolean array to filter out the 2D cells
        twod_idx = grid['nod_type'] == 1  # TODO: get value out of wrapper
        maxk_idx = grid['nodk'][twod_idx]-1
        imaxk = grid['imaxk'][maxk_idx]
        jmaxk = grid['jmaxk'][maxk_idx]
        m = (grid['nodm'][twod_idx] - 1)*imaxk
        n = (grid['nodn'][twod_idx] - 1)*jmaxk

        # TODO: handle 1D stuff correctly
        size = imaxk  # grid['imaxk'][grid['nodk']-1]
        mc = m + size/2.0
        nc = n + size/2.0

        points = np.c_[mc.ravel() * grid['dxp'] + grid['x0p'],
                       nc.ravel() * grid['dyp'] + grid['y0p']]
        self.points = points
        # create array with values
        values = np.zeros_like(mc.ravel())
        # create an interpolation function
        # replace L.values with a an array of size points,nvar to interpolate
        L = scipy.interpolate.LinearNDInterpolator(points, values)
        s = np.s_[
            grid['y0p']:grid['y1p']:complex(0, grid['jmax']),
            grid['x0p']:grid['x1p']:complex(0, grid['imax'])
        ]

        Y, X = np.mgrid[s]
        transform = (float(grid['x0p']),  # xmin
                     float(grid['dxp']),  # xmax
                     0,            # for rotation
                     float(grid['y0p']),
                     0,
                     float(grid['dyp']))
        self.transform = transform
        self.wkt = grid['wkt']
        self.interpolation_ready = True
        return X, Y, L

    def update_grids(self):
        """Preprocess some stuff that only needs to be done once.

        Needs to be run when quad_grid or dps is updated.
        """
        grid = self.grid
        quad_grid = grid['quad_grid']
        dps = grid['dps']
        logger.debug('quad grid shape: %r' % (str(quad_grid.shape)))
        logger.debug('dps shape: %r' % (str(dps.shape)))
        # Sometimes quad_grid.mask is False instead of a table... (model Miami)
        # TODO: investigate more
        if quad_grid.mask.__class__.__name__ == 'bool_':
            mask = np.logical_or.reduce([dps < -9000, ])
        else:
            mask = np.logical_or.reduce(
                [quad_grid.mask, dps < -9000])  # 4 seconds
        if 'quad_grid_dps_mask' in grid:
            del self.grid['quad_grid_dps_mask']
        self.grid['quad_grid_dps_mask'] = mask

    def get(self, layer, interpolate='nearest', from_disk=False, **kwargs):
        """
        layer: choose from waterlevel, waterheight, dps, uc,
          sg, quad_grid, infiltration,
          interception, soil, crop, maxdepth, arrival

        from_disk: read grids.nc instead of memory, kwargs must
        contain kw 'layers':
          duifp-duifp:maxdepth

        NOTE: maxdepth and arrival REQUIRE the from_disk method.

        TODO: disk cache when using from_disk
        """
        def generate_hash(path, layer_slug):
            return '%r-%r-%r' % (
                layer_slug, os.path.getctime(path), os.path.getmtime(path))

        grid = None
        if from_disk:
            logger.debug('Memory from file...')
            layer_slug = kwargs['layers'].split(':')[0]
            logger.debug(layer_slug)
            grid_path = os.path.join(
                config.DATA_DIR, '3di', layer_slug, 'grids.nc')

            if ('file-memory' in self.grid and
                    self.grid['file-memory'] == generate_hash(
                        grid_path, layer_slug)):

                # already loaded
                # if a new file is placed in the same location, it is
                # not detected!!
                logger.debug('already loaded from file into memory')
                grid = self.grid
            else:
                # load file into memory
                logger.debug('loading file into memory')
                nc = Dataset(grid_path, 'r', format='NETCDF3_CLASSIC')
                grid = {}
                grid['dsnop'] = nc.variables['dsnop'].getValue()[0]
                grid['wkt'] = ''.join(nc.variables['wkt'])
                grid['dps'] = nc.variables['dps'][:].copy()
                grid['maxdepth'] = nc.variables['maxdepth'][:].copy()
                grid['arrival'] = nc.variables['arrival'][:].copy()

                grid['x0p'] = nc.variables['x0p'].getValue()[0]
                grid['y0p'] = nc.variables['y0p'].getValue()[0]
                grid['x1p'] = nc.variables['x1p'].getValue()[0]
                grid['y1p'] = nc.variables['y1p'].getValue()[0]
                grid['dxp'] = nc.variables['dxp'].getValue()[0]
                grid['dyp'] = nc.variables['dyp'].getValue()[0]

                grid['file-memory'] = generate_hash(grid_path, layer_slug)
                grid['layer-slug'] = layer_slug  # needed for getcapabilities
                self.grid = grid
                nc.close()

        if grid is None:
            if not self.grid:
                logger.info(
                    'Initializing grids (is normally already done,'
                    ' unless some server error)')
                return None  # Crashes, try again later!
            grid = self.grid

        # try to get parameters from request
        srs = kwargs.get("srs")
        bbox_str = kwargs.get("bbox")
        if bbox_str:
            bbox = [float(x) for x in bbox_str.split(",")]
        else:
            bbox = None
        logger.debug('bbox: %r' % str(bbox))
        height = int(kwargs.get("height", "0"))
        width = int(kwargs.get("width", "0"))
        # multiply the slicing stepsize with 'fast'.
        fast = float(kwargs.get("fast", "1.4"))

        if all([srs, bbox, height, width]):
            logger.debug("slicing and dicing")

            # TODO rename dst/src to map, slice, grid
            src_srs = osgeo.osr.SpatialReference()
            src_srs.ImportFromEPSGA(int(srs.split(':')[1]))
            dst_srs = osgeo.osr.SpatialReference()

            if 'wkt' in grid and grid['wkt']:
                dst_srs.ImportFromWkt(grid["wkt"])
                if (dst_srs.GetAuthorityCode("PROJCS") == '28992'
                        and not dst_srs.GetTOWGS84()):
                    logger.error(
                        "Check WKT for TOWGS84 string! Je weet tog ;-)")
            else:
                logger.warning(
                    'Something is probably wrong with the wkt (%r),'
                    ' taking default 28992.' % grid['wkt'])
                dst_srs.ImportFromEPSGA(28992)

            src2dst = osgeo.osr.CoordinateTransformation(src_srs, dst_srs)

            (xmin, ymin, xmax, ymax) = bbox
            # Beware: the destination is NOT rectangular, so we need to
            # recalculate the bbox.
            x0_dst, y0_dst, _ = src2dst.TransformPoint(xmin, ymin)
            x1_dst, y1_dst, _ = src2dst.TransformPoint(xmax, ymin)
            x2_dst, y2_dst, _ = src2dst.TransformPoint(xmin, ymax)
            x3_dst, y3_dst, _ = src2dst.TransformPoint(xmax, ymax)
            xmin_dst = min([x0_dst, x1_dst, x2_dst, x3_dst])
            xmax_dst = max([x0_dst, x1_dst, x2_dst, x3_dst])
            ymin_dst = min([y0_dst, y1_dst, y2_dst, y3_dst])
            ymax_dst = max([y0_dst, y1_dst, y2_dst, y3_dst])

            # lookup required slice
            xmin_src, ymin_src = (grid['x0p'], grid['y0p'])
            xmax_src, ymax_src = (grid['x1p'], grid['y1p'])
            dx_src, dy_src = (grid['dxp'], grid['dyp'])
            x_src = np.arange(xmin_src, xmax_src, dx_src)
            y_src = np.arange(ymin_src, ymax_src, dy_src)
            # Lookup indices of plotted grid
            # this can be done faster with a calculation
            dps_shape = grid['dps'].shape
            x_start = min(
                max(bisect.bisect(x_src, xmin_dst) - 1, 0),
                dps_shape[1]-1)
            x_end = min(
                max(bisect.bisect(x_src, xmax_dst) + 1, 0),
                dps_shape[1])
            y_start = min(
                max(bisect.bisect(y_src, ymin_dst) - 1, 0),
                dps_shape[0]-1)
            y_end = min(
                max(bisect.bisect(y_src, ymax_dst) + 1, 0),
                dps_shape[0])
            # lookup resolution: restricted to make it faster for big images
            x_step = max(
                trunc(fast * (x_end - x_start)) // min(width, 1200),
                1)
            y_step = max(
                trunc(fast * (y_end - y_start)) // min(height, 800),
                1)
            num_pixels = ((y_end - y_start) // y_step *
                          (x_end - x_start) // x_step)
            logger.debug(
                'Slice: y=%d,%d,%d x=%d,%d,%d width=%d height=%d, pixels=%d'
                % (y_start, y_end, y_step, x_start, x_end, x_step,
                   width, height, num_pixels))
            S = np.s_[y_start:y_end:y_step, x_start:x_end:x_step]

            # Compute transform for sliced grid
            transform = (
                grid["x0p"] + dx_src*x_start,
                dx_src*x_step,
                0,
                grid["y0p"] + grid["dyp"]*y_start,
                0,
                grid["dyp"]*y_step
            )

        else:
            logger.debug("couldn't find enough info in %s", kwargs)
            S = np.s_[:, :]
            transform = self.transform

        if layer == 'waterlevel' or layer == 'waterheight':
            dps = grid['dps'][S].copy()

            dps[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            quad_grid = grid['quad_grid'][S]
            mask = grid['quad_grid_dps_mask'][S]
            s1 = self.grid['s1'].copy()
            vol1 = self.grid['vol1']

            if interpolate == 'nearest':
                # useless option
                waterheight = s1[quad_grid.filled(0)]
            else:
                # Here comes the 'Martijn interpolatie'.
                L = self.L
                if L is None:
                    logger.warn("Interpolation data not available")
                X, Y = self.X[S], self.Y[S]
                try:
                    # Kaapstad gives IndexError
                    volmask = (vol1 == 0)[quad_grid]
                    L.values = np.ascontiguousarray(s1[:, np.newaxis])
                    waterheight = L(X, Y)
                    # now mask the waterlevels where we did not compute
                    # or where mask of the
                    mask = np.logical_or.reduce(
                        [np.isnan(waterheight), mask, volmask])
                    waterheight = np.ma.masked_array(waterheight, mask=mask)
                except IndexError:
                    # Fallback to nearest
                    # Kaapstad:
                    # IndexError: index 1085856568 is out of bounds
                    # for size 16473
                    logger.error(
                        'Interpolation crashed, falling back to nearest.')
                    waterheight = s1[quad_grid.filled(0)]
                    # Log everything
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    for line in traceback.format_exception(
                            exc_type, exc_value, exc_traceback):
                        logger.debug(line)

            if layer == 'waterlevel':
                waterlevel = waterheight - (-dps)

                # Gdal does not know about masked arrays, so we
                # transform to an array with a nodatavalue
                array = np.ma.masked_array(
                    waterlevel, mask=mask).filled(DELTARES_NO_DATA)
                container = rasters.NumpyContainer(
                    array, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            elif layer == 'waterheight':
                waterlevel = waterheight

                # Strange: nodatavalue becomes 0, which is undesirable
                # for getprofile
                array = np.ma.masked_array(waterlevel, mask=mask).filled(-dps)
                container = rasters.NumpyContainer(
                    array, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)

            return container
        elif layer == 'dps':
            dps = grid['dps'][S].copy()

            dps[dps == self.grid['dsnop']] = DELTARES_NO_DATA
            container = rasters.NumpyContainer(
                dps, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'uc':
            quad_grid = grid['quad_grid'][S]
            uc = grid['uc']

            uc_norm = np.sqrt(np.sum(uc**2, axis=0))
            assert uc_norm.shape[0] != 2, "wrong sum dimension"

            container = rasters.NumpyContainer(
                uc_norm[quad_grid], transform, self.wkt)
            return container
        elif layer == 'sg':
            dps = grid['dps'][S].copy()
            quad_grid = grid['quad_grid'][S]
            sg = grid['sg']
            groundwater_depth = -dps - sg[quad_grid]
            # A trick to hold all depths inside model, 0's are filtered out.
            groundwater_depth[np.ma.less_equal(groundwater_depth, 0.01)] = 0.01

            groundwater_depth[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                groundwater_depth, transform, self.wkt,
                nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'sg_abs':
            dps = grid['dps'][S].copy()
            quad_grid = grid['quad_grid'][S]
            sg = grid['sg']
            groundwater_level = sg[quad_grid]

            groundwater_level[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                groundwater_level, transform, self.wkt,
                nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'quad_grid':
            quad_grid = grid['quad_grid'][S]
            container = rasters.NumpyContainer(
                quad_grid, transform, self.wkt)
            return container
        elif layer == 'infiltration':
            dps = grid['dps'][S].copy()
            g = grid['infiltrationrate'][S].copy()

            g[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'interception':
            dps = grid['dps'][S].copy()
            g = grid['maxinterception'][S].copy()

            g[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'soil':
            dps = grid['dps'][S].copy()
            g = grid['soiltype'][S].copy()

            g[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'crop':
            dps = grid['dps'][S].copy()
            g = grid['croptype'][S].copy()

            g[dps == self.grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=DELTARES_NO_DATA)
            return container
        elif layer == 'maxdepth':
            if not from_disk:
                return None  # does not work!

            a = grid['maxdepth'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            a[dps == grid['dsnop']] = DELTARES_NO_DATA

            # Strange stuff: no data value is not handled correctly in
            # preprocessing
            a[a > 10000] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=DELTARES_NO_DATA)
            return container

        elif layer == 'arrival':
            if not from_disk:
                return None  # does not work!

            a = grid['arrival'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            a[dps == grid['dsnop']] = DELTARES_NO_DATA

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=DELTARES_NO_DATA)
            return container

        else:
            raise NotImplemented("working on it")

    def get_raw(self, layer):
        """testing"""
        return self.grid[layer]

    def get_pandas(self, key):
        return self.pandas.get(key, None)

    def __init__(self, sub_port=5558):
        self.sub_port = sub_port
        # When updating, let 'get' function wait
        self.is_updating = BoundedSemaphore(1)

        self.transform = None
        # continuously fill data
        self.loaded_model = None
        self.grid = {}
        # define an interpolation function
        # use update indices to update these variables
        self.L = None
        self.X = None
        self.Y = None
        self.interpolation_ready = False

        self.pandas = {}

        self.thread = None
        self.make_listener(sub_port)  # Listen to model messages
