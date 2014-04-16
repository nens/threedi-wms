
from mmi import send_array, recv_array
from gislib import rasters
from scipy import ndimage

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np
import string
import random
from netCDF4 import Dataset

import time  # stopwatch
import osgeo.osr
import bisect
import sys
import traceback
import os

from server import config

from threading import BoundedSemaphore
from math import trunc

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

        #logger.debug(message_data.grid['y0p'].shape)
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
            self.message_data.grid['nFlowElem2dBounds'])  # Apparently WITH boundary nodes

    def dump_nc(self, var_name, var_type, dimensions, unit, values=None):
        logger.debug('dumping %s...' % var_name)
        if values is None:
            values = self.message_data.grid[var_name]
        self.v = self.ncfile.createVariable(var_name, var_type, dimensions)
        if len(unit) == 0:
            self.v[:] = values
        elif len(unit) == 1:
            # if isinstance(values, str):
            #     self.v[:] = values.split()
            # else:
            self.v[:] = values
        elif len(unit) == 2:
            self.v[:, :] = values  #self.message_data.grid[var_name]

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
        # A flag to notify the thread that it should finish up and exit
        self.kill_received = False

    def run(self):
        """run the thread"""
        message_data = self.message_data
        socket = self.socket
        while not self.kill_received:
            arr, metadata = recv_array(socket)
            logger.info("got msg {}".format(metadata))

            if metadata['action'] == 'reset':
                logger.debug('Resetting grid data...')
                message_data.grid = {}
                message_data.interpolation_ready = False
            elif metadata['action'] == 'update':
                logger.debug('Updating grid data [%s]' % metadata['name'])
                if 'model' in metadata:
                    restarted = metadata['name'] == 't1' and metadata['sim_time_seconds'] < 0.1
                    if metadata['model'] != message_data.loaded_model or restarted:
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

                if (all([v in message_data.grid for v in DEPTH_VARS]) and 
                    metadata['name'] in DEPTH_VARS):

                    logger.debug('Update grids after receiving dps or quad_grid...')
                    message_data.update_grids()
                    logger.debug('Update grids finished.')

                # check update indices
                if (all([v in message_data.grid for v in UPDATE_INDICES_VARS]) and 
                    metadata['name'] in UPDATE_INDICES_VARS):

                    logger.debug('Update indices...')
                    message_data.X, message_data.Y, message_data.L = message_data.calc_indices()
                    logger.debug('Update indices finished.')
            # elif metadata['action'] == 'postprocess':
            #     logger.debug('Post processing...')

            elif metadata['action'] == 'dump':
                output_filename = metadata['output_filename']
                path_nc = os.path.join(metadata['input_path'], 'subgrid_map.nc')

                logger.debug('Dump: checking other threads...')
                if i_am_the_boss(output_filename):
                    nc_dump = NCDump(output_filename, message_data) # TODO: with statement
                    nc_dump.dump_nc('wkt', 'S1', ('i', ), '-', list(message_data.grid['wkt']))
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
                    #nc_dump.dump_nc('quad_grid', 'i4', ('x', 'y', ), '-')
                    #nc_dump.dump_nc('quad_grid_dps_mask', 'i1', ('x', 'y', ), '-')
                    #nc_dump.dump_nc('vol1', 'f4', ('nFlowElem2', ), '-')

                    #nc_dump.dump_nc('nt', 'f8', (), '-')  #testing

                    if os.path.exists(path_nc):

                        grid = message_data.grid
                        L = message_data.L
                        X, Y = message_data.X, message_data.Y

                        with Dataset(path_nc) as dataset:

                            # Set base variables
                            nodatavalue = 1e10
                            dps = grid['dps'].copy()
                            dps[dps == grid['dsnop']] = nodatavalue  # Set the Deltares no data value.
                            quad_grid = grid['quad_grid']  #.filled(0)  # Temp fix error for from_disk
                            mask = grid['quad_grid_dps_mask']
                            vol1 = grid['vol1']


                            # Arrival times
                            nt = grid['nt'].item()  # this timestep
                            dt = grid['dt'].item()  # timestep size

                            s1 = dataset.variables['s1'][:].filled(-9999)
                            time_array = np.ones(grid['dps'].shape) * -9999

                            arrival_times = [0, 3600, 3600*2, 3600*3, 3600*4, 3600*5]
                            s1_agg = []
                            for i, arrival_time in enumerate(arrival_times[:-1]):
                                if nt > arrival_times[i] // dt:
                                    logger.debug('adding %r..' % arrival_times[i])
                                    s1_agg.append(s1[arrival_times[i]//dt:min(arrival_times[i+1]//dt, nt), :].max(0))
                            if nt > arrival_times[-1]:
                                s1_agg.append(s1[arrival_times[-1]:nt, :].max(0))
                            logger.debug('s1 agg: %r' % len(s1_agg))

                            for i, s1_time in enumerate(s1_agg):
                                logger.debug(' processing s1 time interval: %d' % i)
                                start_time = time.time()

                                # Here comes the 'Martijn interpolatie'.
                                L.values = np.ascontiguousarray(s1_time[:,np.newaxis])
                                s1_waterlevel = L(X, Y)
                                # now mask the waterlevels where we did not compute
                                # or where mask of the
                                s1_mask = np.logical_or.reduce([np.isnan(s1_waterlevel), mask])
                                s1_waterlevel = np.ma.masked_array(s1_waterlevel, mask=s1_mask)

                                s1_waterdepth = s1_waterlevel - (-dps)

                                # Gdal does not know about masked arrays, so we transform to an array with 
                                #  a nodatavalue
                                array = np.ma.masked_array(s1_waterlevel, mask=s1_mask).filled(nodatavalue)

                                time_array[np.logical_and(time_array==-9999, array>0)] = i + 1

                            nc_dump.dump_nc('arrival', 'f4', ('x', 'y'), 'm', time_array)


                            # Max waterlevel. Somehow this part influences
                            # "Arrival times". So do not move.
                            s1_max = dataset.variables['s1'][:].max(0)

                            volmask = (vol1 == 0)[quad_grid]  # Kaapstad gives IndexError
                            L.values = np.ascontiguousarray(s1_max[:,np.newaxis])
                            waterlevel = L(X, Y)
                            # now mask the waterlevels where we did not compute
                            # or where mask of the
                            mask = np.logical_or.reduce([np.isnan(waterlevel), mask, volmask])
                            waterlevel = np.ma.masked_array(waterlevel, mask=mask)

                            maxdepth = waterlevel - (-dps)
                            nc_dump.dump_nc('maxdepth', 'f4', ('x', 'y'), 'm', maxdepth)

                    else:
                        logger.error('No subgrid_map file found at %r, skipping' % path_nc)

                    nc_dump.close()  
            else:
                logger.debug('Got an unknown message: %r' % metadata)


class MessageData(object):
    """
    Container for model message data
    """
    def make_listener(self, sub_port):
        """make a socket that waits for new data in a thread"""
        subsock = ctx.socket(zmq.SUB)
        subsock.connect("tcp://localhost:{port}".format(port=sub_port))
        subsock.setsockopt(zmq.SUBSCRIBE,b'')
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
        #logger.info('nodk: %r' % grid['nodk'] )
        #import pdb; pdb.set_trace()

        # twod_idx is a boolean array to filter out the 2D cells
        twod_idx = grid['nod_type'] == 1  # TODO: get value out of wrapper
        imaxk = grid['imaxk'][grid['nodk'][twod_idx]-1]
        jmaxk = grid['jmaxk'][grid['nodk'][twod_idx]-1]
        m = (grid['nodm'][twod_idx] - 1)*imaxk
        n = (grid['nodn'][twod_idx] - 1)*jmaxk

        # TODO: handle 1D stuff correctly
        #m = (grid['nodm']-1)*grid['imaxk'][grid['nodk']-1]
        #n = (grid['nodn']-1)*grid['jmaxk'][grid['nodk']-1]
        size = imaxk  # grid['imaxk'][grid['nodk']-1]
        mc = m + size/2.0
        nc = n + size/2.0

        points = np.c_[mc.ravel() * grid['dxp'] + grid['x0p'] ,nc.ravel() * grid['dyp'] + grid['y0p']]
        self.points = points
        # create array with values
        values = np.zeros_like(mc.ravel())
        # create an interpolation function
        # replace L.values with a an array of size points,nvar to interpolate
        L = scipy.interpolate.LinearNDInterpolator(points, values)
        s = np.s_[
            grid['y0p']:grid['y1p']:complex(0,grid['jmax']),
            grid['x0p']:grid['x1p']:complex(0,grid['imax'])
        ]
        #self.x, self.y = np.ogrid[s]
        Y , X = np.mgrid[s]
        transform= (float(grid['x0p']),  # xmin
                    float(grid['dxp']), # xmax
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
        try:
            mask = np.logical_or.reduce([quad_grid.mask, dps<-9000])  # 4 seconds
            if 'quad_grid_dps_mask' in grid:
                del grid['quad_grid_dps_mask']
            grid['quad_grid_dps_mask'] = mask
        except ValueError:
            # In rare cases, we get an ValueError. Origin unknown
            logger.error('ValueError in messages.py. TODO: investigate')

    def get(self, layer, interpolate='nearest', from_disk=False, **kwargs):
        """
        layer: choose from waterlevel, waterheight, dps, uc, 
          sg, quad_grid, infiltration,
          interception, soil, crop, maxdepth, arrival

        from_disk: read grids.nc instead of memory, kwargs must contain kw 'layers':
          duifp-duifp:maxdepth

        NOTE: maxdepth and arrival REQUIRE the from_disk method.

        TODO: disk cache when using from_disk
        """
        grid = None
        if from_disk:
            logger.debug('Memory from file...')
            layer_slug = kwargs['layers'].split(':')[0]
            logger.debug(layer_slug)

            if 'file-memory' in self.grid and self.grid['file-memory'] == layer_slug:
                # already loaded
                # if a new file is placed in the same location, it is not detected!!
                logger.debug('already loaded from file into memory')
                grid = self.grid
            else:
                # load file into memory
                logger.debug('loading file into memory')
                grid_path = os.path.join(config.DATA_DIR, '3di', layer_slug, 'grids.nc')
                nc = Dataset(grid_path, 'r', format='NETCDF3_CLASSIC')
                grid = {}
                grid['dsnop'] = nc.variables['dsnop'].getValue()[0]
                # grid['quad_grid_dps_mask'] = nc.variables['quad_grid_dps_mask'][:]
                # grid['quad_grid'] = np.ma.masked_array(
                #     nc.variables['quad_grid'][:], 
                #     mask=grid['quad_grid_dps_mask'])
                # grid['vol1'] = nc.variables['vol1'][:]
                grid['wkt'] = ''.join(nc.variables['wkt'])
                grid['dps'] = nc.variables['dps'][:].copy()
                #grid['maxlevel'] = nc.variables['maxlevel'][:]
                grid['maxdepth'] = nc.variables['maxdepth'][:].copy()
                grid['arrival'] = nc.variables['arrival'][:].copy()

                grid['x0p'] = nc.variables['x0p'].getValue()[0]
                grid['y0p'] = nc.variables['y0p'].getValue()[0]
                grid['x1p'] = nc.variables['x1p'].getValue()[0]
                grid['y1p'] = nc.variables['y1p'].getValue()[0]
                grid['dxp'] = nc.variables['dxp'].getValue()[0]
                grid['dyp'] = nc.variables['dyp'].getValue()[0]

                grid['file-memory'] = layer_slug
                self.grid = grid
                # grid['imax'] = nc.variables['imax'][:]
                # grid['jmax'] = nc.variables['jmax'][:]
                # grid['imaxk'] = nc.variables['imaxk'][:]
                # grid['jmaxk'] = nc.variables['jmaxk'][:]

                # grid['nodm'] = nc.variables['nodm'][:]
                # grid['nodn'] = nc.variables['nodn'][:]
                # grid['nodk'] = nc.variables['nodk'][:]
                # grid['nod_type'] = nc.variables['nod_type'][:]

                # # testing
                # grid['nt'] = nc.variables['nt'].getValue()[0]
                nc.close()

        if grid is None:
            if not self.grid:
                logger.info('Initializing grids (is normally already done, unless some server error)')
                return None  # Crashes, try again later!
            grid = self.grid
        time_start = time.time()

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
        fast = float(kwargs.get("fast", "1.4"))  # multiply the slicing stepsize with 'fast'.

        if all([srs, bbox, height, width]):
            logger.debug("slicing and dicing")

            # TODO rename dst/src to map, slice, grid
            src_srs = osgeo.osr.SpatialReference()
            src_srs.ImportFromEPSGA(int(srs.split(':')[1]))
            dst_srs = osgeo.osr.SpatialReference()
            logger.debug("wkt %r" % grid["wkt"])
            if 'wkt' in grid and grid['wkt']:
                dst_srs.ImportFromWkt(grid["wkt"])
                if dst_srs.GetAuthorityCode("PROJCS") == '28992' and not dst_srs.GetTOWGS84():
                    logger.error("Check WKT for TOWGS84 string! Je weet tog ;-)")
            else:
                logger.warning(
                    'Something is probably wrong with the wkt (%r), taking default 28992.' % grid['wkt'])
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
            logger.debug(xmin_src)
            logger.debug(xmax_src)
            logger.debug(dx_src)
            x_src = np.arange(xmin_src, xmax_src, dx_src)
            y_src = np.arange(ymin_src, ymax_src, dy_src)
            # Lookup indices of plotted grid
            # this can be done faster with a calculation
            dps_shape = grid['dps'].shape
            x_start = min(max(bisect.bisect(x_src, xmin_dst) - 1, 0), dps_shape[1]-1)
            x_end = min(max(bisect.bisect(x_src, xmax_dst) + 1, 0), dps_shape[1])
            y_start = min(max(bisect.bisect(y_src, ymin_dst) - 1, 0), dps_shape[0]-1)
            y_end = min(max(bisect.bisect(y_src, ymax_dst) + 1, 0), dps_shape[0])
            # and lookup required resolution
            # /2 is to reduce aliasing=hi quality. *2 is for speed
            x_step = max(trunc(fast * (x_end - x_start)) // width, 1)
            y_step = max(trunc(fast * (y_end - y_start)) // height, 1)
            logger.debug('Slice: y=%d,%d,%d x=%d,%d,%d width=%d height=%d' % (
                y_start, y_end, y_step, x_start, x_end, x_step, width, height))
            S = np.s_[y_start:y_end:y_step, x_start:x_end:x_step]
            #S = np.s_[:,:]
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
            S = np.s_[:,:]
            transform = self.transform
        # logger.debug('transform: %s' % str(transform))
            
        if layer == 'waterlevel' or layer == 'waterheight':
            nodatavalue = 1e10
            dps = grid['dps'][S].copy()
            dps[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.
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
                #L = scipy.interpolate.LinearNDInterpolator(self.points, s1)
                # scipy interpolate does not deal with masked arrays
                # so we set waterlevels to nan where volume is 0
                #s1[vol1 == 0] = np.nan
                #s1 = np.where(vol1 == 0, -self.grid['dmax'], s1)
                try:
                    volmask = (vol1 == 0)[quad_grid]  # Kaapstad gives IndexError
                    L.values = np.ascontiguousarray(s1[:,np.newaxis])
                    waterheight = L(X, Y)
                    # now mask the waterlevels where we did not compute
                    # or where mask of the
                    mask = np.logical_or.reduce([np.isnan(waterheight), mask, volmask])
                    waterheight = np.ma.masked_array(waterheight, mask=mask)
                except IndexError:
                    # Fallback to nearest
                    # Kaapstad:
                    # IndexError: index 1085856568 is out of bounds for size 16473
                    logger.error('Interpolation crashed, falling back to nearest.')
                    waterheight = s1[quad_grid.filled(0)]
                    # Log everything
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    for line in traceback.format_exception(
                        exc_type, exc_value, exc_traceback):
                        logger.debug(line)

            if layer == 'waterlevel':
                waterlevel = waterheight - (-dps)

                # Gdal does not know about masked arrays, so we transform to an array with 
                #  a nodatavalue
                array = np.ma.masked_array(waterlevel, mask=mask).filled(nodatavalue)
                container = rasters.NumpyContainer(array, transform, self.wkt, 
                                                   nodatavalue=nodatavalue)
            elif layer == 'waterheight':
                waterlevel = waterheight

                # Strange: nodatavalue becomes 0, which is undesirable for getprofile
                array = np.ma.masked_array(waterlevel, mask=mask).filled(-dps)
                container = rasters.NumpyContainer(array, transform, self.wkt, 
                                                   nodatavalue=nodatavalue)


            return container
        elif layer == 'dps':
            dps = grid['dps'][S].copy()

            nodatavalue = 1e10
            dps[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

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
                groundwater_depth, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'quad_grid':
            quad_grid = grid['quad_grid'][S]
            container = rasters.NumpyContainer(
                quad_grid, transform, self.wkt)
            return container
        elif layer == 'infiltration':
            dps = grid['dps'][S].copy()
            g = grid['infiltrationrate'][S].copy()

            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'interception':
            dps = grid['dps'][S].copy()
            g = grid['maxinterception'][S].copy()

            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'soil':
            dps = grid['dps'][S].copy()
            g = grid['soiltype'][S].copy()

            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'crop':
            dps = grid['dps'][S].copy()
            g = grid['croptype'][S].copy()

            nodatavalue = 1e10
            g[dps == self.grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            container = rasters.NumpyContainer(
                g, transform, self.wkt, nodatavalue=nodatavalue)
            return container
        elif layer == 'maxdepth':
            if not from_disk:
                return None  # does not work!

            a = grid['maxdepth'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            nodatavalue = 1e10
            a[dps == grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            # Strange stuff: no data value is not handled correctly in preprocessing
            a[a > 10000] = nodatavalue  

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=nodatavalue)
            return container

        elif layer == 'arrival':
            if not from_disk:
                return None  # does not work!

            logger.debug(np.amin(a))
            logger.debug(np.amax(a))
            a = grid['arrival'][S].copy()
            dps = grid['dps'][S].copy()
            wkt = grid['wkt']

            nodatavalue = 1e10
            a[dps == grid['dsnop']] = nodatavalue  # Set the Deltares no data value.

            container = rasters.NumpyContainer(
                a, transform, wkt, nodatavalue=nodatavalue)
            return container

        else:
            raise NotImplemented("working on it")

    def get_raw(self, layer):
        """testing"""
        return self.grid[layer]

    def __init__(self, sub_port=5558):
        #self.req_port = req_port
        self.sub_port = sub_port
        self.is_updating = BoundedSemaphore(1)  # When updating, let 'get' function wait

        self.transform = None
        # continuously fill data
        #self.data = {}
        self.loaded_model = None
        self.grid = {}
        # define an interpolation function
        # use update indices to update these variables
        self.L = None
        #self.x = None
        #self.y = None
        self.X = None
        self.Y = None
        self.interpolation_ready = False

        self.thread = None
        self.make_listener(sub_port) # Listen to model messages
