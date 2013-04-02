threedi-wms
===========

A flexible implementation of parts of the wms standard using Flask for
the web stuff. The Flask Blueprint system is used to disclose a variety
of datasources.

The main purpose of the library is visualization of fast and accurate
3di flooding calculations.

Features:
    - Celery worker for long lasting tasks such as building bathymetry pyramids
    - File and only file based configuration
    - Modular setup employing Flask's Blueprint system


Prerequisities
--------------

For the build to work a number of system libraries is required::

  $ libhdf5-serial-dev # Not sure about this one.
  $ libnetcdf-dev


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
    
Go to localhost:5000/3di/demo to see the server in action.
