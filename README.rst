threedi-wms
===========

A flexible implementation of parts of the wms standard using Flask for
the web stuff. The Flask Blueprint system is used to disclose a variety
of datasources.

The main purpose of the library is visualization of fast and accurate
3di flooding calculations.

Features:
    - Memory messages for viewing live 3Di data
    - NetCDF source for time series
    - Celery worker for long lasting tasks such as building bathymetry pyramids
    - File and only file based configuration
    - Modular setup employing Flask's Blueprint system


Prerequisities
--------------

For the build to work a number of system libraries is required::

  $ libhdf5-serial-dev # Not sure about this one.
  $ libnetcdf-dev
  $ libfreetype6-dev

GDAL 1.9.1 or higher. The server will NOT work correctly with a lower version
(grid is misplaced).


Running
-------

Production::

    $ bin/gunicorn 'server.app:build_app(req_port=5557,sub_port=5558)' -w 1 -b 0.0.0.0:5000

Note: You must have the threedi_server configured at given request/subscribe ports in order to use memory messages.

Development::

    $ bin/flask


Installation of the wms server
------------------------------
First the basic steps::

    $ python bootstrap.py
    $ bin/buildout

Put your datasets in var/data/3di/<myfolder>. The name of the folder will
be the name of the layer in the request. In the folder should be a .nc
file and a .tif or .asc bathymetry file. .tif performs better than .asc
in the preparation step, but after that there is no difference since
the bathymetry is cached in a pyramid oject on the filesystem. Create
the tif from the bathimetry .asc file with gdal::

    $ gdal_translate mybathimetry.asc myresult.tif

Start the server and the task processor using::

    $ bin/supervisord
    
Go to localhost:5000/3di/demo to see the server in action. To see
the requests for timeseries and profiles along a line, check the
developertools of the browser while single clicking a point on the map
(timeseries) or a single click followed by a double click on some other
point (profile along a line) on the demo page.

Examples of requests for contours and quanties directly from the netcdf::

    http://localhost:5000/3di/data?request=getcontours&layers=purmer2
    http://localhost:5000/3di/data?request=getquantity&layers=purmer2&
        quantity=dep&time=0&decimals=2


Dutch
-----

Wanneer threedi-wms draait, zijn er 2 lopende processen. Een die handelt wms / url aanvragen af (flask). De ander luistert naar ZMQ voor inkomende data (server/messages.py, class Listener). Beide geinitieerd vanuit server/app.py. De app.py wordt op een dergelijke manier opgestart: $ src/threedi-wms/bin/gunicorn "server.app:build_app(req_port=5556,sub_port=5558)" -w 1 -b 0.0.0.0:5000

- In het begin als je threedi-wms opstart is hij 'leeg'

- Het rekenhart gaat grids sturen. Wanneer bepaalde grids binnen zijn, gaat threedi-wms afgeleide grids berekenen:                 
         if (all([v in message_data.grid for v in DEPTH_VARS]) and 
                    metadata['name'] in DEPTH_VARS):

- Het 'dump' commando is de uitvoering van "archive".             
        elif metadata['action'] == 'dump':,

- Er zit nog een gekke i_am_the_boss functie in. Dit is omdat meerdere threedi-wms processen aan staan en je wilt maar dat eentje het dump commando gaat uitvoeren.


Rasterinfo (not yet working?)
-----------------------------
The rasterinfo server serves a HTTP layer over gislib functonality

Go to localhost:5000/rasterinfo/ to confirm the service is working. You will see a link to the profile tool

**Profile tool:**
Send a linestring and projection to the server which returns a (json) list of (x, value) pairs where X = distance in projection units from start of line and value = value of the raster at that point.

**Example:** 
`http://localhost:5000/rasterinfo/profile?geom=LINESTRING(570060.51753709%206816367.7101946,568589.10474281%206815374.028827)&srs=EPSG:900913`


