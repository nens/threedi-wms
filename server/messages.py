
from mmi import send_array, recv_array
from gislib import rasters
from scipy import ndimage

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np

import time  # stopwatch
import osgeo.osr
import bisect
import sys
import traceback

from threading import BoundedSemaphore

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
                    message_data.update_indices()
                    logger.debug('Update indices finished.')


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

    def update_indices(self):
        """create all the indices that we need for performance

        These vars probably use a lot of memory.
        """
        # lookup cell centers
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
        self.L = scipy.interpolate.LinearNDInterpolator(points, values)
        s = np.s_[
            grid['y0p']:grid['y1p']:complex(0,grid['jmax']),
            grid['x0p']:grid['x1p']:complex(0,grid['imax'])
        ]
        self.x, self.y = np.ogrid[s]
        self.Y , self.X = np.mgrid[s]
        transform= (float(grid['x0p']),  # xmin
                    float(grid['dxp']), # xmax
                    0,            # for rotation
                    float(grid['y0p']),
                    0,
                    float(grid['dyp']))
        self.transform = transform
        self.wkt = grid['wkt']
        self.interpolation_ready = True

    def update_grids(self):
        """Preprocess some stuff that only needs to be done once.

        Needs to be run when quad_grid or dps is updated.
        """
        grid = self.grid
        quad_grid = grid['quad_grid']
        dps = grid['dps']
        logger.debug('quad grid shape: %r' % (str(quad_grid.shape)))
        logger.debug('dps shape: %r' % (str(dps.shape)))
        mask = np.logical_or.reduce([quad_grid.mask, dps<-9000])  # 4 seconds
        if 'quad_grid_dps_mask' in grid:
            del grid['quad_grid_dps_mask']
        grid['quad_grid_dps_mask'] = mask

    def get(self, layer, interpolate='nearest', **kwargs):
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
            x_src = np.arange(xmin_src, xmax_src, dx_src)
            y_src = np.arange(ymin_src, ymax_src, dy_src)
            # Lookup indices of plotted grid
            # this can be done faster with a calculation
            dps_shape = self.grid['dps'].shape
            x_start = min(max(bisect.bisect(x_src, xmin_dst) - 1, 0), dps_shape[1]-1)
            x_end = min(max(bisect.bisect(x_src, xmax_dst) + 1, 0), dps_shape[1])
            y_start = min(max(bisect.bisect(y_src, ymin_dst) - 1, 0), dps_shape[0]-1)
            y_end = min(max(bisect.bisect(y_src, ymax_dst) + 1, 0), dps_shape[0])
            # and lookup required resolution
            x_step = max((x_end - x_start) // width, 1)
            y_step = max((y_end - y_start) // height, 1)
            logger.debug('Slice: y=%d,%d,%d x=%d,%d,%d width=%d height=%d' % (
                y_start, y_end, y_step, x_start, x_end, x_step, width, height))
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
            uc = grid['uc'][S]
            container = rasters.NumpyContainer(
                uc, transform, self.wkt)
            return container
        elif layer == 'quad_grid':
            quad_grid = grid['quad_grid'][S]
            container = rasters.NumpyContainer(
                quad_grid, transform, self.wkt)
            return container
        else:
            raise NotImplemented("working on it")

    def get_raw(self, layer):
        """testing"""
        return self.grid[layer]

    def __init__(self, req_port=5556, sub_port=5558):
        self.req_port = req_port
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
        self.x = None
        self.y = None
        self.X = None
        self.Y = None
        self.interpolation_ready = False

        self.thread = None
        self.make_listener(sub_port) # Listen to model messages
