from mmi import send_array, recv_array
from gislib import rasters
from scipy import ndimage

import scipy.interpolate
import zmq
import logging
import threading
import numpy as np

import time  # stopwatch

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
                    if metadata['model'] != message_data.loaded_model:
                        # New model detected
                        logging.info('New model detected: %r' % metadata['model'])
                        message_data.loaded_model = metadata['model']
                        message_data.data = {}
                        message_data.grid = None
                message_data.data[metadata['name']] = arr
        thread = threading.Thread(target=model_listener,
                                  args=[subsock, message_data]
                                  )
        thread.daemon = True
        thread.start()

    @staticmethod
    def recv_grid(req_port=5556, timeout=5000):
        """connect to the socket to get an updated grid"""
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
        if self._grid is None:
            try:
                self._grid = MessageData.recv_grid(req_port=self.req_port)
                logging.debug("Grid received")
                self.update_indices()
                logging.debug("Indices created")
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

    def get(self, layer, interpolate='nearest'):
        grid = self.grid
        time_start = time.time()
        if layer == 'waterlevel':
            logging.debug('start waterlevel...')
            dps = grid["dps"]
            quad_grid = grid['quad_grid']
            logging.debug('or reduce...')
            logging.debug('time %2f' % (time.time() - time_start))
            mask = np.logical_or.reduce([quad_grid.mask, dps<-9000])  # 4 seconds
            logging.debug('time %2f' % (time.time() - time_start))
            logging.debug('check s1...')
            if 's1' not in self.data:
                logging.info('Requesting init data...')
                self.data = self.recv_grid(req_port=self.req_port)  # triggers init data
            logging.debug('s1...')
            s1 = self.data['s1']
            #logging.debug("shape s1: {}".format(s1.shape))
            #logging.debug("quad_grid, min-max: {} {}".format(quad_grid.min(), quad_grid.max()))
            if interpolate == 'nearest':
                logging.debug('nearest interpolation...')
                logging.debug('time %2f' % (time.time() - time_start))
                waterheight = s1[quad_grid.filled(0)]  # 1.5 seconds
                logging.debug('time %2f' % (time.time() - time_start))
                #logging.debug("s1 : {} {}".format(waterheight.min(), waterheight.max()))
            else:
                logging.debug('linear interpolation...')
                #L = scipy.interpolate.LinearNDInterpolator(self.points, s1)
                self.L.values = np.ascontiguousarray(s1[:,np.newaxis])
                L = self.L
                waterheight = L(self.X, self.Y) 
                mask = np.logical_or(np.isnan(waterheight), mask)
                waterheight = np.ma.masked_array(waterheight, mask=mask)
                #logging.debug("s1 : {} {}".format(waterheight.min(), waterheight.max()))
             
            logging.debug('waterlevel...')   
            waterlevel = waterheight - (-dps)
            #logging.debug("s1  - - dps: {} {}".format(waterlevel.min(), waterlevel.max()))
            logging.debug('masked array...')   
            array = np.ma.masked_array(waterlevel, mask = mask)
            logging.debug('container...')   
            container = rasters.NumpyContainer(array, self.transform, self.wkt)
            logging.debug('time %2f' % (time.time() - time_start))
            return container
        elif layer == 'bathymetry':
            container = rasters.NumpyContainer(grid['dps'], self.transform, self.wkt)
            return container
        elif layer == 'quad_grid':
            container = rasters.NumpyContainer(self.grid["quad_grid"], self.transform, self.wkt)
            return container
        else:
            raise NotImplemented("working on it")

    def __init__(self, req_port=5556, sub_port=5558):
        self.req_port = req_port
        self.sub_port = sub_port

        self.transform = None
        # continuously fill data
        self.data = {}
        self.loaded_model = None
        self._grid = None
        self.grid
        self.make_listener(self, sub_port)

        # define an interpolation function
        # use update indices to update these variables
        self.L = None
        self.x = None
        self.y = None
        self.X = None
        self.Y = None
