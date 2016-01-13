# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
from mmi import recv_array
from gislib import rasters

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np
import string
import random
from netCDF4 import Dataset

import time
import osgeo.osr
import bisect
import sys
import traceback
import os
import json

from server import config, utils
from status import StateReporter

from threading import BoundedSemaphore
from math import trunc


logger = logging.getLogger('')

ctx = zmq.Context()

UPDATE_INDICES_VARS = [
    'nod_type', 'imaxk', 'nodk', 'jmaxk', 'nodm', 'nodn',
    'dxp', 'x0p', 'dyp', 'y0p', 'x1p', 'y1p', 'imax', 'jmax', 'wkt']
DEPTH_VARS = [
    'dps', 'quad_grid']


def i_am_the_boss(filename, timeout_seconds=5):
    """Make sure I can write filename on my own"""

    # A file metaphore mechanism: only one thread should take the message
    def touch(fname, id_string):
        with file(fname, 'w') as f:
            os.utime(fname, (time.time(), time.time()))
            f.write(id_string)

    def verify(fname, id_string):
        with file(fname, 'r') as f:
            id_string_file = f.readline()
        return id_string == id_string_file

    def generate_id(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    filename_busy = filename + '.busy'
    if (os.path.exists(filename_busy) and
            os.path.getmtime(filename_busy) > time.time() - timeout_seconds):
        return False

    # try to claim filename
    id_string = generate_id()
    touch(filename_busy, id_string)
    # race condition check:
    return verify(filename_busy, id_string)


class NCDump(object):
    def __init__(self, output_filename, message_data):

        self.message_data = message_data

        # dump the data in netcdf
        logger.debug('Ready to go.')
        logger.debug('Dumping to netcdf...')
        self.ncfile = Dataset(output_filename, 'w', format='NETCDF3_CLASSIC')

        x_dim = self.ncfile.createDimension(
            'x', self.message_data.grid['quad_grid'].shape[0])
        y_dim = self.ncfile.createDimension(
            'y', self.message_data.grid['quad_grid'].shape[1])
        i_dim = self.ncfile.createDimension(
            'i', None)   # random index, for wkt
        i_dim = self.ncfile.createDimension(
            'k', self.message_data.grid['imaxk'].shape[0])   # no idea what it is, needed for imaxk, jmaxk
        flow_elem_dim = self.ncfile.createDimension(
            'nFlowElem1', self.message_data.grid['nFlowElem1d']+
            self.message_data.grid['nFlowElem2d'])  # Apparently no boundary nodes
        flow_elem2_dim = self.ncfile.createDimension(
            'nFlowElem2',
            self.message_data.grid['nFlowElem1d'] +
            self.message_data.grid['nFlowElem1dBounds'] +
            self.message_data.grid['nFlowElem2d'] +
            # Apparently WITH boundary nodes
            self.message_data.grid['nFlowElem2dBounds'])

    def dump_nc(self, var_name, var_type, dimensions, unit, values=None):
        """In some weird cases, this function can crash with a RuntimeError
        from NETCDF: RuntimeError: NetCDF: Operation not allowed in define mode

        Thus it is preferred that the function runs in a try/except.
        """
        logger.debug('dumping %s...' % var_name)
        if values is None:
            values = self.message_data.grid[var_name]
        self.v = self.ncfile.createVariable(var_name, var_type, dimensions)
        logger.info('dimensions %s' % str(dimensions))
        logger.info('len(unit) %d' % len(unit))
        if len(unit) == 0:
            self.v[:] = values
        elif len(unit) == 1:
            self.v[:] = values
        elif len(unit) == 2:
            self.v[:, :] = values

        self.v.units = unit
        self.v.standard_name = var_name

    def close(self):
        logger.debug('Closing...')
        self.ncfile.sync()
        self.ncfile.close()
        logger.debug('Done')


class Listener(threading.Thread):
    def __init__(self, socket, message_data, *args, **kwargs):
        self.socket = socket
        self.message_data = message_data
        threading.Thread.__init__(self, *args, **kwargs)
        # a flag to notify the thread that it should finish up and exit
        self.kill_received = False
        self.reporter = StateReporter()

    def reset_grid_data(self):
        logger.debug(
            'Resetting grid data...',
            extra={'subgrid_id': self.reporter.redis_key})
        for k in self.message_data.grid.keys():
            del self.message_data.grid[k]  # try to save memory
        self.message_data.interpolation_ready = False
        self.message_data.pandas = {}

    def _run(self):
        """run the thread"""
        message_data = self.message_data
        socket = self.socket
        while not self.kill_received:
            logger.debug(
                '(a) number of busy workers: %s' %
                self.reporter.get_busy_workers(),
                extra={'subgrid_id': self.reporter.redis_key})
            arr, metadata = recv_array(socket)
            # now it is busy
            self.reporter.set_busy()
            # N.B.: to simulate the wms_busy state uncomment the following
            # line, but do not commit it, never!
            # time.sleep(random.uniform(0.0, 0.5))
            logger.debug(
                '(b) number of busy workers: %s' %
                self.reporter.get_busy_workers(),
                extra={'subgrid_id': self.reporter.redis_key})
            logger.debug(
                'time in seconds wms is considered busy: %s' %
                str(self.reporter.busy_duration),
                extra={'subgrid_id': self.reporter.redis_key})
            if metadata['action'] == 'reset':
                self.reset_grid_data()
            elif metadata['action'] == 'update':
                self.reporter.set_timestep(metadata['sim_time_seconds'])
                logger.debug(
                    'Updating grid data [%s]' % metadata['name'],
                    extra={'subgrid_id': self.reporter.redis_key})
                if 'model' in metadata:
                    restarted = metadata['name'] == 't1' and \
                        metadata['sim_time_seconds'] < 0.1
                    if metadata['model'] != message_data.loaded_model or \
                            restarted:
                        # Since working with 'reset', this part probably never
                        # occur anymore.

                        # New model detected
                        logger.info(
                            'New model detected: %r (old=%r)' % (
                                metadata['model'], message_data.loaded_model),
                            extra={'subgrid_id': self.reporter.redis_key})

                        # Double reset algorithm.
                        message_data.grid = {}
                        message_data.interpolation_ready = False
                        message_data.loaded_model = metadata['model']

                # Update new grid
                if arr.dtype.kind == 'S':
                    # String, for wkt
                    message_data.grid[metadata['name']] = ''.join(arr)
                else:
                    if 'bbox' in metadata:
                        logger.debug("BBOXED update")
                        x0, x1, y0, y1 = metadata['bbox']
                        message_data.grid[metadata['name']][y0:y1, x0:x1] = arr
                    else:
                        # normal case
                        message_data.grid[metadata['name']] = arr.copy()

                # Receive one of the DEPTH_VARS and all DEPTH_VARS are complete
                if (all([v in message_data.grid for v in DEPTH_VARS]) and
                        metadata['name'] in DEPTH_VARS):

                    if 'bbox' in metadata:
                        logger.debug(
                            'Update grids using bbox after receiving '
                            'dps or quad_grid...',
                            extra={'subgrid_id': self.reporter.redis_key})
                        message_data.update_grids_bbox(metadata['bbox'])
                    else:
                        logger.debug(
                            'Update grids after receiving dps or quad_grid...',
                            extra={'subgrid_id': self.reporter.redis_key})
                        message_data.update_grids()
                    logger.debug(
                        'Update grids finished.',
                        extra={'subgrid_id': self.reporter.redis_key})

                # check update indices
                if (all([v in message_data.grid for v in UPDATE_INDICES_VARS])
                        and metadata['name'] in UPDATE_INDICES_VARS):
                    logger.debug(
                        'Update indices...',
                        extra={'subgrid_id': self.reporter.redis_key})
                    message_data.X, message_data.Y, message_data.L = \
                        message_data.calc_indices()
                    logger.debug(
                        'Update indices finished.',
                        extra={'subgrid_id': self.reporter.redis_key})
            elif metadata['action'] == 'update-pandas':
                logger.debug(
                    'Update pandas data [%s]...', metadata['name'],
                    extra={'subgrid_id': self.reporter.redis_key})
                # TODO: in case of weir, delete unused variables.
                message_data.pandas[metadata['name']] = json.loads(
                    metadata['pandas_json'])

            elif metadata['action'] == 'dump':
                output_filename = metadata['output_filename']
                logger.debug(
                    'Dump: checking other threads...',
                    extra={'subgrid_id': self.reporter.redis_key})
                filename_failed = output_filename + '.failed'
                if os.path.exists(filename_failed):
                    os.remove(filename_failed)
                if i_am_the_boss(output_filename):
                    # TODO: with statement
                    nc_dump = NCDump(output_filename, message_data)
                    nc_dump.dump_nc('wkt', 'S1', ('i', ), '-',
                                    list(message_data.grid['wkt']))
                    nc_dump.dump_nc('x0p', 'f8', (), '-')
                    nc_dump.dump_nc('y0p', 'f8', (), '-')
                    nc_dump.dump_nc('x1p', 'f8', (), '-')
                    nc_dump.dump_nc('y1p', 'f8', (), '-')
                    nc_dump.dump_nc('dxp', 'f8', (), '-')
                    nc_dump.dump_nc('dyp', 'f8', (), '-')
                    nc_dump.dump_nc('imax', 'i4', (), '-')
                    nc_dump.dump_nc('jmax', 'i4', (), '-')
                    nc_dump.dump_nc('imaxk', 'i4', ('k', ), '-')  #
                    nc_dump.dump_nc('jmaxk', 'i4', ('k', ), '-')
                    nc_dump.dump_nc('nodm', 'i4', ('nFlowElem2', ), '-')
                    nc_dump.dump_nc('nodn', 'i4', ('nFlowElem2', ), '-')
                    nc_dump.dump_nc('nodk', 'i4', ('nFlowElem2', ), '-')
                    nc_dump.dump_nc('nod_type', 'i4', ('nFlowElem2', ), '-')

                    nc_dump.dump_nc('dsnop', 'f4', (), '-')
                    nc_dump.dump_nc('dps', 'f4', ('x', 'y', ), '-')

                    try:
                        nc_dump.close()
                    except:
                        # I don't know when nc_dump will fail, but if it fails,
                        # it is probably here.
                        with file(filename_failed, 'w') as f:
                            f.write('I failed...')
                    # So others can see we are finished.
                    os.remove(output_filename + '.busy')
            else:
                logger.debug(
                    'Got an unknown message: %r' % metadata,
                    extra={'subgrid_id': self.reporter.redis_key})
            # set this worker to not busy
            self.reporter.set_not_busy()
            self.reporter.handle_busy_flag()

    def run(self):
        """Run the thread fail-safe"""
        while not self.kill_received:
            try:
                self._run()
            except:
                logger.exception(
                    'An unknown exception occured, subgrid_id: %s' %
                    self.reporter.redis_key)
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
        self.subgrid_id = thread.reporter.redis_key
        # In a hook of the website: thread.kill_received = True

    def stop_listener(self):
        if self.thread and self.thread.isAlive:
            logger.debug(
                "Killing listener in thread {}".format(self.thread),
                extra={'subgrid_id': self.subgrid_id})
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
        twod_idx = grid['nod_type'] == 1
        maxk_idx = grid['nodk'][twod_idx]-1
        imaxk = grid['imaxk'][maxk_idx]
        jmaxk = grid['jmaxk'][maxk_idx]
        m = (grid['nodm'][twod_idx] - 1)*imaxk
        n = (grid['nodn'][twod_idx] - 1)*jmaxk

        size = imaxk
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
        """Precalculate quad_grid_dps_mask that only needs to be calculated once.

        Needs to be run when quad_grid or dps is updated.
        """
        grid = self.grid
        quad_grid = grid['quad_grid']
        dps = grid['dps']
        logger.debug(
            'quad grid shape: %r' % (str(quad_grid.shape)),
            extra={'subgrid_id': self.subgrid_id})
        logger.debug(
            'dps shape: %r' % (str(dps.shape)),
            extra={'subgrid_id': self.subgrid_id})
        # Sometimes quad_grid.mask is False instead of a table... (model Miami)
        # TODO: investigate more
        if quad_grid.mask.__class__.__name__ == 'bool_':
            mask = utils.logical_or_reduce([dps < -9000, ])
        else:
            # 4 seconds
            mask = utils.logical_or_reduce([quad_grid.mask, dps < -9000])
        if 'quad_grid_dps_mask' in grid:
            del self.grid['quad_grid_dps_mask']
        self.grid['quad_grid_dps_mask'] = mask

    def update_grids_bbox(self, bbox):
        """Update quad_grid_dps_mask using bbox

        bbox format: [x0, x1, y0, y1]
        """
        if 'quad_grid_dps_mask' not in self.grid:
            logger.debug(
                "Calling update_grids instead of update_grids_bbox.",
                extra={'subgrid_id': self.subgrid_id})
            return self.update_grids()
        x0, x1, y0, y1 = bbox
        quad_grid = self.grid['quad_grid'][y0:y1, x0:x1]
        dps = self.grid['dps'][y0:y1, x0:x1]
        logger.debug(
            'quad grid bbox shape: %r' % (str(quad_grid.shape)),
            extra={'subgrid_id': self.subgrid_id})
        logger.debug(
            'dps bbox shape: %r' % (str(dps.shape)),
            extra={'subgrid_id': self.subgrid_id})
        # Sometimes quad_grid.mask is False instead of a table... (model Miami)
        # TODO: investigate more
        if quad_grid.mask.__class__.__name__ == 'bool_':
            mask = utils.logical_or_reduce([dps < -9000, ])
        else:
            mask = utils.logical_or_reduce([quad_grid.mask, dps < -9000])
        self.grid['quad_grid_dps_mask'][y0:y1, x0:x1] = mask

    def get(self, layer, interpolate='nearest', from_disk=False, **kwargs):
        """
        layer: choose from waterlevel, waterheight, dps, uc,
          sg, quad_grid, infiltration,
          interception, soil, crop, maxdepth, arrival

        from_disk: read grids.nc instead of memory, kwargs must contain kw
            'layers': duifp-duifp:maxdepth

        NOTE: maxdepth and arrival REQUIRE the from_disk method.
        """
        def generate_hash(path, layer_slug):
            return '%r-%r-%r' % (
                layer_slug, os.path.getctime(path), os.path.getmtime(path))

        grid = None
        if from_disk:
            logger.debug(
                'Memory from file...',
                extra={'subgrid_id': self.subgrid_id})
            layer_slug = kwargs['layers'].split(':')[0]
            logger.debug(layer_slug)
            grid_path = os.path.join(config.DATA_DIR, '3di', layer_slug,
                                     'grids.nc')

            if ('file-memory' in self.grid and
                    self.grid['file-memory'] == generate_hash(
                    grid_path, layer_slug)):

                # already loaded
                # if a new file is placed in the same location, it is not
                # detected!!
                logger.debug(
                    'already loaded from file into memory',
                    extra={'subgrid_id': self.subgrid_id})
                grid = self.grid
            else:
                # load file into memory
                logger.debug(
                    'loading file into memory',
                    extra={'subgrid_id': self.subgrid_id})
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
                    'Initializing grids (is normally already done, unless '
                    'some server error)',
                    extra={'subgrid_id': self.subgrid_id})
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
            logger.debug(
                "slicing and dicing",
                extra={'subgrid_id': self.subgrid_id})

            src_srs = osgeo.osr.SpatialReference()
            src_srs.ImportFromEPSGA(int(srs.split(':')[1]))
            dst_srs = osgeo.osr.SpatialReference()
            if 'wkt' in grid and grid['wkt']:
                dst_srs.ImportFromWkt(grid["wkt"])
                if dst_srs.GetAuthorityCode("PROJCS") == '28992' and \
                        not dst_srs.GetTOWGS84():
                    logger.error(
                        "Check WKT for TOWGS84 string! Je weet tog ;-)")
            else:
                logger.warning(
                    'Something is probably wrong with the wkt, taking '
                    'default 28992.',
                    extra={'subgrid_id': self.subgrid_id})
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
            x_start = \
                min(max(bisect.bisect(x_src, xmin_dst) - 1, 0), dps_shape[1]-1)
            x_end = \
                min(max(bisect.bisect(x_src, xmax_dst) + 1, 0), dps_shape[1])
            y_start = \
                min(max(bisect.bisect(y_src, ymin_dst) - 1, 0), dps_shape[0]-1)
            y_end = \
                min(max(bisect.bisect(y_src, ymax_dst) + 1, 0), dps_shape[0])
            # lookup resolution: restricted to make it faster for big images
            x_step = max(
                trunc(fast * (x_end - x_start)) // min(width, 1200), 1)
            y_step = max(
                trunc(fast * (y_end - y_start)) // min(height, 800), 1)
            num_pixels = \
                (y_end - y_start) // y_step * (x_end - x_start) // x_step
            logger.debug(
                'Slice: y=%d,%d,%d x=%d,%d,%d width=%d height=%d, pixels=%d' %
                (y_start, y_end, y_step, x_start, x_end, x_step, width, height,
                 num_pixels),
                extra={'subgrid_id': self.subgrid_id})
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
            logger.debug(
                "couldn't find enough info in %s", kwargs,
                extra={'subgrid_id': self.subgrid_id})
            S = np.s_[:, :]
            transform = self.transform

        if layer == 'waterlevel' or layer == 'waterheight':
            nodatavalue = 1e10
            dps = grid['dps'][S].copy()
            # Set the Deltares no data value.
            dps[dps == self.grid['dsnop']] = nodatavalue
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
                    logger.warn(
                        "Interpolation data not available",
                        extra={'subgrid_id': self.subgrid_id})
                X, Y = self.X[S], self.Y[S]
                try:
                    # Kaapstad gives IndexError
                    volmask = (vol1 == 0)[quad_grid]
                    L.values = np.ascontiguousarray(s1[:, np.newaxis])
                    waterheight = L(X, Y)
                    # now mask the waterlevels where we did not compute
                    # or where mask of the
                    mask = utils.logical_or_reduce(
                        [np.isnan(waterheight), mask, volmask])
                    waterheight = np.ma.masked_array(waterheight, mask=mask)
                except IndexError:
                    # Fallback to nearest
                    # Kaapstad:
                    # IndexError: index 1085856568 is out of bounds for size
                    #   16473
                    logger.error(
                        'Interpolation crashed, falling back to nearest.',
                        extra={'subgrid_id': self.subgrid_id})
                    waterheight = s1[quad_grid.filled(0)]
                    # Log everything
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    for line in traceback.format_exception(
                            exc_type, exc_value, exc_traceback):
                        logger.debug(line)

            if layer == 'waterlevel':
                waterlevel = waterheight - (-dps)

                # Gdal does not know about masked arrays, so we transform to
                # an array with a nodatavalue
                array = np.ma.masked_array(waterlevel, mask=mask).filled(
                    nodatavalue)
                container = rasters.NumpyContainer(array, transform, self.wkt,
                                                   nodatavalue=nodatavalue)
            elif layer == 'waterheight':
                waterlevel = waterheight

                # Strange: nodatavalue becomes 0, which is undesirable for
                # getprofile
                array = np.ma.masked_array(waterlevel, mask=mask).filled(-dps)
                container = rasters.NumpyContainer(array, transform, self.wkt,
                                                   nodatavalue=nodatavalue)

            return container
        elif layer == 'dps':
            dps = grid['dps'][S].copy()

            # Set the Deltares no data value.
            nodatavalue = 1e10
            dps[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                dps, transform, self.wkt, nodatavalue=nodatavalue)
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

            # Set the Deltares no data value.
            nodatavalue = 1e10
            groundwater_depth[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                groundwater_depth, transform, self.wkt,
                nodatavalue=nodatavalue)
            return container
        elif layer == 'sg_abs':
            dps = grid['dps'][S].copy()
            quad_grid = grid['quad_grid'][S]
            sg = grid['sg']
            groundwater_level = sg[quad_grid]
            # A trick to hold all depths inside model, 0's are filtered out.
            # groundwater_depth[np.ma.less_equal(groundwater_depth,
            #                                    0.01)] = 0.01

            # Set the Deltares no data value.
            nodatavalue = 1e10
            groundwater_level[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                groundwater_level, transform, self.wkt,
                nodatavalue=nodatavalue)
            return container
        elif layer == 'quad_grid':
            quad_grid = grid['quad_grid'][S]
            container = rasters.NumpyContainer(
                quad_grid, transform, self.wkt)
            return container
        elif layer == 'infiltration':
            dps = grid['dps'][S].copy()
            g = grid['infiltrationrate'][S].copy()

            # Set the Deltares no data value.
            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'interception':
            dps = grid['dps'][S].copy()
            g = grid['maxinterception'][S].copy()

            # Set the Deltares no data value.
            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'soil':
            dps = grid['dps'][S].copy()
            g = grid['soiltype'][S].copy()

            # Set the Deltares no data value.
            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'crop':
            dps = grid['dps'][S].copy()
            g = grid['croptype'][S].copy()

            # Set the Deltares no data value.
            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'maxdepth':
            if not from_disk:
                return None  # does not work!

            a = grid['maxdepth'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            # Set the Deltares no data value.
            nodatavalue = 1e10
            a[dps == grid['dsnop']] = nodatavalue

            # Strange stuff: no data value is not handled correctly in
            # preprocessing
            a[a > 10000] = nodatavalue

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=nodatavalue)
            return container

        elif layer == 'arrival':
            if not from_disk:
                return None  # does not work!

            a = grid['arrival'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            # Set the Deltares no data value.
            nodatavalue = 1e10
            a[dps == grid['dsnop']] = nodatavalue

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=nodatavalue)
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
