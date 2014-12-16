"""Moved the dump_data stuff from messages.py here."""

import logging
import os
import random
import string
import time

from netCDF4 import Dataset
import numpy as np
import osgeo.osr

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DELTARES_NO_DATA = 1e10
NO_DATA = -9999


def dump_data(output_filename, input_directory, message_data):
    path_nc = os.path.join(input_directory, 'subgrid_map.nc')

    logger.debug('Dump: checking other threads...')
    filename_failed = output_filename + '.failed'
    if os.path.exists(filename_failed):
        os.remove(filename_failed)

    if not i_am_the_boss(output_filename):
        # Some other thread is already writing
        return

    nc_dump = NCDump(output_filename, message_data)
    nc_dump.dump_nc('wkt', 'S1', ('i', ), '-', list(message_data.grid['wkt']))
    nc_dump.dump_nc('x0p', 'f8', (), '-')
    nc_dump.dump_nc('y0p', 'f8', (), '-')
    nc_dump.dump_nc('x1p', 'f8', (), '-')
    nc_dump.dump_nc('y1p', 'f8', (), '-')
    nc_dump.dump_nc('dxp', 'f8', (), '-')
    nc_dump.dump_nc('dyp', 'f8', (), '-')
    nc_dump.dump_nc('imax', 'i4', (), '-')
    nc_dump.dump_nc('jmax', 'i4', (), '-')
    nc_dump.dump_nc('imaxk', 'i4', ('k', ), '-')
    nc_dump.dump_nc('jmaxk', 'i4', ('k', ), '-')
    nc_dump.dump_nc('nodm', 'i4', ('nFlowElem2', ), '-')
    nc_dump.dump_nc('nodn', 'i4', ('nFlowElem2', ), '-')
    nc_dump.dump_nc('nodk', 'i4', ('nFlowElem2', ), '-')
    nc_dump.dump_nc('nod_type', 'i4', ('nFlowElem2', ), '-')
    nc_dump.dump_nc('dsnop', 'f4', (), '-')
    nc_dump.dump_nc('dps', 'f4', ('x', 'y', ), '-')

    if os.path.exists(path_nc):
        grid = message_data.grid
        L = message_data.L
        X, Y = message_data.X, message_data.Y

        with Dataset(path_nc) as dataset:
            # Set base variables
            dps = grid['dps'].copy()
            dps[dps == grid['dsnop']] = DELTARES_NO_DATA
            # Temp fix error for from_disk
            quad_grid = grid['quad_grid']
            mask = grid['quad_grid_dps_mask']
            vol1 = grid['vol1']

            # Arrival times this timestep
            nt = int(grid['nt'].item())
            # timestep size seconds
            dt = int(grid['dtmax'].item())

            s1 = dataset.variables['s1'][:].filled(NO_DATA)
            time_array = np.ones(grid['dps'].shape) * NO_DATA

            arrival_times = [0, 3600, 3600 * 2, 3600 * 3, 3600 * 5, 3600 * 10]
            s1_agg = []

            for i, arrival_time in enumerate(arrival_times[:-1]):
                if nt > arrival_times[i] // dt:
                    logger.debug('adding %r (%r:%r)..' % (
                        arrival_times[i], arrival_times[i] // dt,
                        min(arrival_times[i + 1] // dt, nt)))
                    s1_agg.append(s1[
                        arrival_times[i] // dt:
                        min(arrival_times[i + 1] // dt, nt), :].max(0))

            if nt > arrival_times[-1] // dt:
                logger.debug('adding max...')
                s1_agg.append(s1[arrival_times[-1]//dt:nt, :].max(0))
            logger.debug('s1 agg: %r' % len(s1_agg))

            for i, s1_time in enumerate(s1_agg):
                logger.debug(' processing s1 time interval: %d' % i)

                # Here comes the 'Martijn interpolatie'.
                L.values = np.ascontiguousarray(s1_time[:, np.newaxis])
                s1_waterlevel = L(X, Y)

                # now mask the waterlevels where we did not compute or
                # where mask of the
                s1_mask = np.logical_or.reduce([np.isnan(s1_waterlevel), mask])
                s1_waterlevel = np.ma.masked_array(s1_waterlevel, mask=s1_mask)
                s1_waterdepth = s1_waterlevel - (-dps)

                # Gdal does not know about masked arrays, so we
                # transform to an array with a nodatavalue
                array = np.ma.masked_array(
                    s1_waterdepth, mask=s1_mask).filled(DELTARES_NO_DATA)

                time_array[
                    np.logical_and(time_array == NO_DATA, array > 0)] = i + 1

            nc_dump.dump_nc('arrival', 'f4', ('x', 'y'), 'm', time_array)

            arrival_filename = os.path.join(os.path.dirname(
                output_filename), 'arrival.tif')
            dump_geotiff(arrival_filename, time_array)

            # Max waterlevel. Somehow this part influences "Arrival
            # times". So do not move.
            s1_max = dataset.variables['s1'][:].max(0)

            # Kaapstad gives IndexError
            volmask = (vol1 == 0)[quad_grid]
            L.values = np.ascontiguousarray(s1_max[:, np.newaxis])
            waterlevel = L(X, Y)

            # now mask the waterlevels where we did not compute or
            # where mask of the
            mask = np.logical_or.reduce([np.isnan(waterlevel), mask, volmask])
            waterlevel = np.ma.masked_array(waterlevel, mask=mask)

            maxdepth = np.maximum(waterlevel - (-dps), 0)
            nc_dump.dump_nc('maxdepth', 'f4', ('x', 'y'), 'm', maxdepth)

            maxdepth_filename = os.path.join(os.path.dirname(
                output_filename), 'maxdepth.tif')
            # seems like interpolations take place to the masked value
            # sometimes outside the working area
            dump_geotiff(
                maxdepth_filename,
                np.ma.masked_greater(maxdepth, 10000).filled(
                    fill_value=NO_DATA))

    else:
        logger.error('No subgrid_map file found at %r, skipping' % path_nc)

    try:
        nc_dump.close()
    except:
        # I don't know when nc_dump will fail, but if it fails, it is
        # probably here.
        with file(filename_failed, 'w') as f:
            f.write('I failed...')

    # So others can see we are finished.
    os.remove(output_filename + '.busy')


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

        self.ncfile.createDimension(
            'x', self.message_data.grid['quad_grid'].shape[0])  # x dim
        self.ncfile.createDimension(
            'y', self.message_data.grid['quad_grid'].shape[1])  # y dim
        self.ncfile.createDimension(
            'i', None)   # random index, for wkt

        self.ncfile.createDimension(
            # no idea what it is, needed for imaxk, jmaxk
            'k', self.message_data.grid['imaxk'].shape[0])

        self.ncfile.createDimension(  # flow_elem_dim
            'nFlowElem1', self.message_data.grid['nFlowElem1d'] +
            # Apparently no boundary nodes
            self.message_data.grid['nFlowElem2d'])
        self.ncfile.createDimension(
            'nFlowElem2',
            self.message_data.grid['nFlowElem1d'] +
            self.message_data.grid['nFlowElem1dBounds'] +
            self.message_data.grid['nFlowElem2d'] +
            # Apparently WITH boundary nodes
            self.message_data.grid['nFlowElem2dBounds'])

    def dump_nc(self, var_name, var_type, dimensions, unit, values=None):
        """In some weird cases, this function can crash with a RuntimeError
        from NETCDF: RuntimeError: NetCDF: Operation not allowed in
        define mode

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


def dump_geotiff(output_filename, values):
    """Dump data to geotiff

    Currently only works for dem_hhnk.tif / 5m because of all the constants.
    """
    # Import libs
    logger.info("Writing [%s] geotiff..." % output_filename)
    origin_x, origin_y = 98970.000000000000000, 553410.000000000000000
    height, width = values.shape

    # Create gtif
    driver = osgeo.gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_filename, width, height, 1,
                           osgeo.gdal.GDT_Float32, ['COMPRESS=DEFLATE', ])

    raster = np.flipud(values)

    # top left x, w-e pixel resolution, rotation, top left y,
    # rotation, n-s pixel resolution
    dst_ds.SetGeoTransform([origin_x, 5, 0, origin_y, 0, -5])

    # set the reference info
    srs = osgeo.osr.SpatialReference()
    srs.ImportFromEPSG(28992)

    dst_ds.SetProjection(srs.ExportToWkt())

    # write the band
    dst_ds.GetRasterBand(1).WriteArray(raster)
    dst_ds.GetRasterBand(1).SetNoDataValue(NO_DATA)
