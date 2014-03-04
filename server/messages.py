
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

from threading import BoundedSemaphore

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# global zmq context 
ctx = zmq.Context()


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
        message_data.init_grids()
        socket = self.socket
        while not self.kill_received:
            arr, metadata = recv_array(socket)
            logger.info("got msg {}".format(metadata))
            if 'model' in metadata:
                restarted = metadata['name'] == 't1' and metadata['sim_time_seconds'] < 0.1
                if metadata['model'] != message_data.loaded_model or restarted:
                    # New model detected
                    logger.info('New model detected: %r (old=%r)' % (
                        metadata['model'], message_data.loaded_model))
                    #message_data.loaded_model = metadata['model']
                    #message_data.grid = message_data.recv_grid()
                    message_data.init_grids()
            if metadata['name'] in message_data.grid:
                del message_data.grid[metadata['name']]  # saves memory
            message_data.grid[metadata['name']] = arr
            logger.debug('I have data for: %r' % message_data.grid.keys())
            if metadata['name'] == 'dps' or metadata['name'] == 'quad_grid':
                logger.debug('Update grids after receiving dps or quad_grid...')
                message_data.update_grids()

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

    def recv_grid(self, req_port=5556, timeout=20000):
        """connect to the socket to get an updated grid
        TODO: not nice that this is different than the listener

        TODO: check timeout
        """
        # We don't have a message format for this yet
        # We could keep this socket open.
        req = ctx.socket(zmq.REQ)
        # Blocks until connection is found
        logger.info("Getting new grid from socket {}".format(req) )
        req.connect("tcp://localhost:{port}".format(port=req_port))
        # Wait at most 5 seconds
        req.setsockopt(zmq.RCVTIMEO, timeout)
        # try 10 times
        req.send_json({"action": "send init"})
        #grid = req.recv_pyobj()
        try:
            grid = req.recv_pyobj()
            return grid
        except zmq.error.Again:
            logger.exception("Grid not received")
        finally:
            req.close()

    def update_indices(self):
        """create all the indices that we need for performance

        These vars probably use a lot of memory.
        """
        # lookup cell centers
        grid = self.grid
        m = (grid['nodm']-1)*grid['imaxk'][grid['nodk']-1]
        n = (grid['nodn']-1)*grid['jmaxk'][grid['nodk']-1]
        size = grid['imaxk'][grid['nodk']-1]
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

    def init_grids(self):
        logger.debug('init grids...')
        self.grid = {}
        time_start = time.time()
        logger.debug('receiving grids...')
        new_grid = self.recv_grid(req_port=self.req_port)  # triggers init data
        logger.debug('stopped receiving, now have %s', self.grid.keys())
        if new_grid is not None:
            self.grid = new_grid
            self.loaded_model = self.grid['loaded_model']
            logger.debug('time after receive grid %2f' % (time.time() - time_start))
            self.update_indices()
            logger.debug('time after update indices %2f' % (time.time() - time_start))
            self.update_grids()
            logger.debug('time after update grids %2f' % (time.time() - time_start))
            logger.debug('now have keys: %s' % (', '.join(self.grid.keys())))
        else:
            self.loaded_model = False
            logger.debug('init grids failed.')

    def get(self, layer, interpolate='nearest', **kwargs):
        if not self.grid:
            logger.info('Initializing grids (is normally already done, unless some server error)')
            return None  # Crashes, try again later!
            # new_grid = self.init_grids()
            # if new_grid:
            #     self.grid = new_grid
        # else:
        #     logger.debug('Grids keys %r' % self.grid.keys())
        #     logger.debug('Kwargs %r' % kwargs)
        grid = self.grid
        time_start = time.time()

        # try to get parameters from request

        srs = kwargs.get("srs")
        bbox_str = kwargs.get("bbox")
        if bbox_str:
            bbox = [float(x) for x in bbox_str.split(",")]
        else:
            bbox = None
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
            xmin_dst, ymin_dst, _ = src2dst.TransformPoint(xmin,ymin)
            xmax_dst, ymax_dst, _ = src2dst.TransformPoint(xmax, ymax)

            # lookup required slice
            xmin_src, ymin_src = (grid['x0p'], grid['y0p'])
            xmax_src, ymax_src = (grid['x1p'], grid['y1p'])
            dx_src, dy_src = (grid['dxp'], grid['dyp'])
            x_src = np.arange(xmin_src, xmax_src, dx_src)
            y_src = np.arange(ymin_src, ymax_src, dy_src)
            # Lookup indices of plotted grid
            # this can be done faster with a calculation
            x_start = bisect.bisect(x_src, xmin_dst)
            x_end = bisect.bisect(x_src, xmax_dst)
            y_start = bisect.bisect(y_src, ymin_dst)
            y_end = bisect.bisect(y_src, ymax_dst)
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
        logger.debug('transform: %s' % str(transform))
            
        if layer == 'waterlevel':
            logger.debug('start waterlevel...')
            dps = grid['dps'][S]
            quad_grid = grid['quad_grid'][S]
            mask = grid['quad_grid_dps_mask'][S]
            s1 = self.grid['s1'].copy()
            vol1 = self.grid['vol1']

            if interpolate == 'nearest':
                logger.debug('nearest interpolation...')
                waterheight = s1[quad_grid.filled(0)]
            else:
                L = self.L
                if L is None:
                    logger.warn("Interpolation data not available")
                X, Y = self.X[S], self.Y[S]
                logger.debug('linear interpolation...')
                #L = scipy.interpolate.LinearNDInterpolator(self.points, s1)
                # scipy interpolate does not deal with masked arrays
                # so we set waterlevels to nan where volume is 0
                #s1[vol1 == 0] = np.nan
                #s1 = np.where(vol1 == 0, -self.grid['dmax'], s1)
                volmask = (vol1 == 0)[quad_grid]
                L.values = np.ascontiguousarray(s1[:,np.newaxis])
                waterheight = L(X, Y)
                #logger.debug('%r', waterheight)
                # now mask the waterlevels where we did not compute
                # or where mask of the
                mask = np.logical_or.reduce([np.isnan(waterheight), mask, volmask])
                waterheight = np.ma.masked_array(waterheight, mask=mask)

            logger.debug('waterlevel...')
            waterlevel = waterheight - (-dps)
            logger.debug('masked array...')
            # Gdal does not know about masked arrays, so we transform to an array with 
            #  a nodatavalue
            nodatavalue = 1e10
            array = np.ma.masked_array(waterlevel, mask = mask).filled(nodatavalue)
            logger.debug('container...')
            container = rasters.NumpyContainer(array, transform, self.wkt, 
                                               nodatavalue=nodatavalue)
            return container
        elif layer == 'dps':
            dps = grid['dps'][S]
            logger.debug('bathymetry')
            container = rasters.NumpyContainer(
                dps, transform, self.wkt)
            return container
        elif layer == 'quad_grid':
            quad_grid = grid['quad_grid'][S]
            logger.debug('quad_grid')
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

        self.thread = None
        self.make_listener(sub_port) # Listen to model messages
