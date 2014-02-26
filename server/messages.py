from mmi import send_array, recv_array
from gislib import rasters
from scipy import ndimage

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np

import time  # stopwatch

from threading import BoundedSemaphore

# global zmq context 
ctx = zmq.Context()


class MessageData(object):
    """
    Container for model message data
    """
    @staticmethod
    def make_listener(message_data, sub_port):
        """make a socket that waits for new data in a thread"""
        subsock = ctx.socket(zmq.SUB)
        subsock.connect("tcp://localhost:{port}".format(port=sub_port))
        subsock.setsockopt(zmq.SUBSCRIBE,b'')
        def model_listener(socket, message_data):
            while True:
                arr, metadata = recv_array(socket)
                logging.info("got msg {}".format(metadata))
                if 'model' in metadata:
                    if metadata['model'] != message_data.loaded_model or (
                        metadata['name'] == 't1' and metadata['sim_time_seconds'] < 0.1):
                        # New model detected
                        logging.info('New model detected: %r (old=%r)' % (
                            metadata['model'], message_data.loaded_model))
                        message_data.loaded_model = metadata['model']
                        message_data.grid = {}
                        message_data.init_grids()
                #message_data.data[metadata['name']] = arr
                message_data.is_updating.acquire()
                if metadata['name'] in message_data.grid:
                    del message_data.grid[metadata['name']]  # saves memory
                message_data.grid[metadata['name']] = arr
                logging.debug('I have data for: %r' % message_data.grid.keys())
                if metadata['name'] == 'dps' or metadata['name'] == 'quad_grid':
                    logging.debug('Update grids after receiving dps or quad_grid...')
                    message_data.update_grids()
                message_data.is_updating.release()
        thread = threading.Thread(target=model_listener,
                                  args=[subsock, message_data]
                                  )
        thread.daemon = True
        thread.start()

    @staticmethod
    def recv_grid(req_port=5556, timeout=5000):
        """connect to the socket to get an updated grid
        TODO: not nice that this is different than the listener
        """
        req = ctx.socket(zmq.REQ)
        # Blocks until connection is found
        req.connect("tcp://localhost:{port}".format(port=req_port))
        # Wait at most 5 seconds
        req.setsockopt(zmq.RCVTIMEO, timeout)
        # We don't have a message format
        req.send_json({"action": "send init"})
        try:
            grid = req.recv_pyobj()
            logging.info("Grid received: %r" % grid.keys())
        except zmq.error.Again:
            logging.exception("Grid not received")
            # We don't have a grid, get it later
            # reraise
            raise 

        return grid

    def getgrid(self):
        if not self._grid:
            try:
                self.init_grids()
                # self._grid = MessageData.recv_grid(req_port=self.req_port)
                # logging.debug("Grid received")
                # self.update_indices()
                # self.update_grids()
                # logging.debug("Indices created")
            except zmq.error.Again as e:
                logging.exception("Grid not received")
            
        return self._grid

    def setgrid(self, value):
        self._grid = value

    def delgrid(self):
        del self._grid
    grid = property(getgrid, setgrid, delgrid, "The grid property")

    def update_indices(self):
        """create all the indices that we need for performance"""

        del self.L
        del self.x
        del self.y
        del self.X
        del self.Y

        # lookup cell centers
        grid = self._grid
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
        grid = self._grid
        quad_grid = grid['quad_grid']
        dps = grid['dps']
        logging.debug('quad grid shape: %r' % (str(quad_grid.shape)))
        logging.debug('dps shape: %r' % (str(dps.shape)))
        mask = np.logical_or.reduce([quad_grid.mask, dps<-9000])  # 4 seconds
        if 'quad_grid_dps_mask' in grid:
            del grid['quad_grid_dps_mask']
        grid['quad_grid_dps_mask'] = mask

    def init_grids(self):
        logging.debug('init grids, acquire semaphore...')
        self.is_updating.acquire()
        del self.grid
        time_start = time.time()
        logging.debug('receiving grids...')
        self._grid = self.recv_grid(req_port=self.req_port)  # triggers init data
        self.loaded_model = self._grid['loaded_model']
        logging.debug('time after receive grid %2f' % (time.time() - time_start))
        self.update_indices()
        logging.debug('time after update indices %2f' % (time.time() - time_start))
        self.update_grids()
        logging.debug('time after update grids %2f' % (time.time() - time_start))
        self.is_updating.release()

    def get(self, layer, interpolate='nearest'):
        grid = self.grid

        time_start = time.time()
        if layer == 'waterlevel':
            logging.debug('start waterlevel...')
            self.is_updating.acquire()
            dps = grid['dps']
            quad_grid = grid['quad_grid']
            mask = grid['quad_grid_dps_mask']
            s1 = self.grid['s1']

            if interpolate == 'nearest':
                logging.debug('nearest interpolation...')
                logging.debug('time %2f' % (time.time() - time_start))
                waterheight = s1[quad_grid.filled(0)]  # 2 seconds (all quads!!)
                logging.debug('time %2f' % (time.time() - time_start))
                #logging.debug("s1 : {} {}".format(waterheight.min(), waterheight.max()))
            else:
                logging.debug('linear interpolation...')  # slow!
                logging.debug('time %2f' % (time.time() - time_start))
                #L = scipy.interpolate.LinearNDInterpolator(self.points, s1)
                self.L.values = np.ascontiguousarray(s1[:,np.newaxis])
                L = self.L
                waterheight = L(self.X, self.Y) 
                mask = np.logical_or(np.isnan(waterheight), mask)
                waterheight = np.ma.masked_array(waterheight, mask=mask)
                logging.debug('time %2f' % (time.time() - time_start))
                #logging.debug("s1 : {} {}".format(waterheight.min(), waterheight.max()))
             
            logging.debug('waterlevel...')   
            logging.debug('time %2f' % (time.time() - time_start))
            waterlevel = waterheight - (-dps)  # 0.5 second
            logging.debug('time %2f' % (time.time() - time_start))
            #logging.debug("s1  - - dps: {} {}".format(waterlevel.min(), waterlevel.max()))
            logging.debug('masked array...')   
            logging.debug('time %2f' % (time.time() - time_start))
            array = np.ma.masked_array(waterlevel, mask = mask)
            logging.debug('time %2f' % (time.time() - time_start))
            logging.debug('container...')   
            logging.debug('time %2f' % (time.time() - time_start))
            container = rasters.NumpyContainer(array, self.transform, self.wkt)
            logging.debug('time %2f' % (time.time() - time_start))
            self.is_updating.release()

            return container
        elif layer == 'dps':
            # if 'dps' not in self.grid:
            #     # temp fix
            #     self.init_grids()
            logging.debug('bathymetry')
            logging.debug('time %2f' % (time.time() - time_start))
            container = rasters.NumpyContainer(
                grid['dps'], self.transform, self.wkt)
            logging.debug('time %2f' % (time.time() - time_start))
            return container
        elif layer == 'quad_grid':
            # if 'quad_grid' not in self.grid:
            #     # temp fix
            #     self.init_grids()
            logging.debug('quad_grid')
            logging.debug('time %2f' % (time.time() - time_start))
            container = rasters.NumpyContainer(
                self.grid['quad_grid'], self.transform, self.wkt)
            logging.debug('time %2f' % (time.time() - time_start))
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
        self._grid = {}
        #self.grid

        # define an interpolation function
        # use update indices to update these variables
        self.L = None
        self.x = None
        self.y = None
        self.X = None
        self.Y = None

        self.init_grids()  # doesn't seem to work?
        self.make_listener(self, sub_port)

