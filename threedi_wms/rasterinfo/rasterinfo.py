from gislib.raster import Pyramid
from gislib.vector import MagicLine

from threedi_wms.rasterinfo import config

from osgeo import gdal, osr
from osgeo.gdalconst import GDT_Float32
from shapely import wkt
import numpy as np


def get_profile(wktline, src_epsg=900913, rastersize=512):
    """
    get raster values for pixels under linestring for Pyramid as
    set in PYRAMID_PATH

    :param wktline: WKT linestring for which profile should be extracted
    :param scr_epsg: spatial reference system EPSG code
    :param rastersize: size of longest side of raster subset

    :returns: list with pairs of [cumlength, rastervalue]
    """
    # setup pyramid
    pyramid = Pyramid(config.PYRAMID_PATH)

    # convert linestring to geometric object with shapely
    linestring = wkt.loads(wktline)
    bounds = linestring.bounds
    points = list(linestring.coords)

    # set longest side to fixed size
    width = bounds[2] - bounds[0]
    length = bounds[3] - bounds[1]
    longside = max(width, length)
    if longside == width:
        xsize = rastersize
        cellsize = width / rastersize
        # +1 = ugly hack to trick the system when top = bottom
        ysize = int(length / cellsize) + 1
    else:
        ysize = rastersize
        cellsize = length / rastersize
        xsize = int(width / cellsize) + 1

    # setup epsg
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(src_epsg)

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
    # quick&dirty oplossing to handle nodata values
    values = np.where(values >= 0, values, 6)
    values = map(float, values)

    # make array with distance from origin (x values for graph)
    # NOTE: linestring.length returns different length than QGis
    # maybe related to projection / planar?
    distances = map(float, np.arange(len(values)) *
                    linestring.length / len(values))
    graph_data = [list(a) for a in zip(distances, values)]

    mem_ds = None

    return graph_data
