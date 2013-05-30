from gislib.raster import Pyramid, get_spatial_reference
from gislib.vector import MagicLine

from threedi_wms.rasterinfo import config

from osgeo import gdal
from osgeo.gdalconst import GDT_Float32
from shapely import wkt
import numpy as np
import math


def get_profile(wktline, src_srs=900913, rastersize=512):
    """
    get raster values for pixels under linestring for Pyramid as
    set in PYRAMID_PATH in config file

    :param wktline: WKT linestring for which profile should be extracted
    :param src_srs: spatial reference system EPSG code
    :param rastersize: size of longest side of raster subset

    :returns: list with pairs of [cumlength, rastervalue]
    """
    # setup pyramid
    pyramid = Pyramid(config.PYRAMID_PATH)

    # setup srs
    try:
        srs = get_spatial_reference(src_srs)
    except:
        return "Malformed EPSG code: %s" % (src_srs)

    # convert linestring to geometric object with shapely
    try:
        linestring = wkt.loads(wktline)
    except:
        return "Malformed geometry: %s" % (wktline)
    bounds = linestring.bounds
    points = list(linestring.coords)

    # set longest side to fixed size
    width = bounds[2] - bounds[0]
    length = bounds[3] - bounds[1]
    longside = max(width, length)
    if longside == width:
        xsize = rastersize
        cellsize = width / rastersize
        ysize = int(max(math.ceil(length / cellsize), 1))
    else:
        ysize = rastersize
        cellsize = length / rastersize
        xsize = int(max(math.ceil(width / cellsize), 1))

    # setup dataset in memory based on bounds
    mem_drv = gdal.GetDriverByName('MEM')
    mem_ds = mem_drv.Create(b'', xsize, ysize, 1, GDT_Float32)
    geotransform = (bounds[0], cellsize, 0, bounds[3], 0, -cellsize)
    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(srs.ExportToWkt())
    origin = np.array([[mem_ds.GetGeoTransform()[0],
                        mem_ds.GetGeoTransform()[3]]])

    # warp values from pyramid into mem dataset
    pyramid.warpinto(mem_ds)

    # make magicline from linestring vertices
    magicline = MagicLine(points)
    magicline = magicline.pixelize(cellsize)

    # Determine indices for these points
    indices = tuple(np.uint64((magicline.centers - origin) / cellsize,
                              ).transpose())[::-1]
    values = mem_ds.ReadAsArray()[indices]
    values = map(float, values)

    # make array with distance from origin (x values for graph)
    distances = map(float, np.arange(len(values)) *
                    linestring.length / len(values))
    profile_data = [list(a) for a in zip(distances, values)]

    mem_ds = None

    return profile_data
