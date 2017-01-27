Changelog of threedi-wms
===================================================


0.11 (unreleased)
-----------------

- Upgrade netCDF4 to 1.0.8.

- Base calculation docker image on trusty (14.04) instead of precise (12.04).

- The unit read from te subgrid netcdf for s1 should be "m MSL". This is
  what will be returned in the ``get_response_for_gettimeseries()`` method.

- Uses the actual subgrid_id as a CACHE_KEY_PREFIX now.

- The ``get_response_for_getmap()``-method caches png's now for 45 seconds
  instead of 5 seconds.

- Use the new ``REDIS_[HOST|DB]_*`` variables.

- Simplify ``fetch_subgrid_id()``.

- Apparently the names of ground water level/depth layers have changed in the
  front end. Old names as well as new names are supported: sg /
  ground_water_depth and sg_abs / ground_water_level.

- Add missing link for mmi 0.1.9 in buildout.cfg. Also, fix index url and kgs
  url in buildout.cfg.

- Pin mmi to 0.1.9.

- Fix KeyError bug in log message.

- PEP8.

- Replace np.logical_or.reduce by a simple looping variant that is much
  faster.

- Added csv output format for gettimeseries (see docstring).

- Change the link numbers redis key.

- Where possible, add ``SUBGRID_ID`` to logging statements.

- Made the default settings suitable for docker containers. ``REDIS_HOST`` can
  now be set with an environment variable.

- When fetching the subgrid_id, look for it in an env var called ``SUBGRID_ID``
  first.

- Better logging for GetMap.

- Added KNOWN_VARS to getquantity; this way flow variables are ignored.

- Improve color scheme for infiltration and interception foreground layers.

- Add sentry logging handler.

- Removed archiving Arrival times and Max depth (it is implemented in
  threedi-result now).

- Add logstash logging handler.

- Better colors for infiltration and interception, range of infiltration to
  0-500.

- Fix tests by reverting bootstrap.py to an older version and using our own
  distribute_setup.py.

- Fix negative number of busy wms workers. This can occur sometimes, but should
  always be corrected to 0, because number of busy workers < 0 does not make
  sense. Reason for it to become lower than 0 is unknown.

- Add a timer for measuring whether wms is busy and put the wms busy state in
  redis, so that it is sent to the client.

- Added getmap sg_abs (ground water level).

- Added option messages for GetInfo.

- Added summary for getprofile: for nxt-graph.

- Added statusfile '.failed' to nc_dump functionality.

- Fix for some rare cases (Miami) in messages.

- Fix for logging line in messages for the Archive function.

- Bumped shapely to 1.2.15 because of import error in version 1.2.12.

- Combine quantity requests.

- Throttle getquantity response by only returning the data for the used flow
  link numbers.

- Added redis cache using flask-cache.

- Added Listener thread master fail-safe try/except loop.

- Splitted imaxk index for easier debugging.

- Added soil and crop color map.

- Fixed no data value being picked up from memory read map.

- Tuned messages 'lookup resolution': tuned for better performance with big hd images.

- Use dtmax instead of dt in 'dump', requires threedi-server 0.43 or newer.

- Added sentry/raven connection for service/maintenance team.

- Removed old use_cache option.

- Return empty image when unsupported map is requested using getmap (an error
  will notify Sentry and takes about 300ms).

- Return empty image for getmap when required data is not available in messages.

- Instead of an empty image now an error image layer is displayed when nonexisting
  map layers are requested using getmap

- Added receiving pandas dataframe objects as json: status of pumps, weirs,
  culverts, orifices.

- Added ability to update parts of grids. Needed for threedi-server 0.67.

- New colors for groundwater (brown to green to white-green).

- Updated interception scaling from hmax=0.20 to hmax=0.02

- Better colors for DEM in getmap.

- Fixed config bug (probably caused pyramid failed errors).

- Bugfix gettimeseriess speeds it up by a large amount.


0.10 (2014-04-17)
-----------------

- Added getmap and file-messages for maxdepth and arrival maps.

- Temp fix for rare crashes in messages.py.

- Added crop and soil layers for getmap (TODO: colors).

- Added infiltration and interception layers for getmap.

- Added option fast to getmap: a value of 0.5 will improve the image quality,
  a value of 2 will make it (a lot) faster.

- Working groundwater: mode == 'sg', e.g. duifp:sg

- Working velocity: mode == 'velocity', e.g. duifp:velocity

- Better messages handling, removed old method.

- Now always having alpha channel.

- Added ground water in getprofile.

- Added fallback mechanism if memory messages not yet available.


0.9 (2014-03-06)
----------------

- Implement volume mask and alpha channel for masked arrays.

- Improved getmap messages.

- Getprofile, gettimeseries now also works with messages. Gettimeseries uses
  messages for getting the height and quad cell.

- Removed gettimeseries:timestep option.

- Improved memory messages, ports are configurable in gunicorn: bin/gunicorn 'server.app:build_app(req_port=5557,sub_port=5558)' -w 1 -b 0.0.0.0:5000


0.8 (2014-02-17)
----------------

- Added option maxpoints to gettimeseries: throw away points till you have the max number of points you have :-)

- Add find link for netCDF4 1.0.4 to buildout.cfg.

- New in memory messages (receive numpy arrays through ZMQ sockets).


0.7 (2014-02-05)
----------------

- Added velocity as possible layer.

- Upgrade to latest buildout to fix problems with distribute during
  bootstrap.


0.6 (2013-12-04)
----------------

- Option 'absolute' now also works for other parameters than s1.

- Fixed bug in gettimeseries. We do NOT want max(v, 0) for everything.

- get_quantity now also works for tables q, unorm.

- Added option timeformat=iso/epoch in gettimeseries.


0.5 (2013-10-21)
----------------

- Added option quad, absolute in gettimeseries.


0.4 (2013-10-07)
----------------

Note: you have to delete the whole cache dir, it is not compatible with the
new gislib.

- Upgraded from gislib 0.1.1 to gislib 0.2.8.

- Added hmax option for GetMap depth.

- Use syseggrecipe for buildout sysegg entry.


0.3 (2013-09-03)
----------------

- Bugfix: now explicitly add srs 28992 if no projection info is available.

- Enabled request variables other than s1 in gettimeseries.

- Made water more pretty by adding alpha, requires matplotlib 1.2.0 or higher
  (1.3.0 requires pyparsing >= 1.5.6 which conflicts).

- Pinned matplotlib 1.2.0 (was 1.1.1rc)


0.2 (2013-08-20)
----------------

- Try to get projection information from geotiff, defaults to rijksdriehoek
  (28992).

- You can now also use gettimeseries to get the height of that point.

- Added bias in profile: this is needed for nv.d3.

- Added option "rebuild_static=yes" for getmap.

- Fixed bug for timeseries ('negative depths'). Timeseries now from t0. More
  negative depths solved.

- Added support for Kaapstad (case insensitive) which is in EPSG:22234.

- Update gislib to 0.1.1

- Added 2 decimals to getprofile.

- Changed response for getprofile to fit nv.d3.

- Added waterlevel and bathymetry to getprofile (bathymetry transposed to 0).


0.1 (2013-07-15)
----------------

- Add profile functionality. See /3di/demo, click, then double click and watch
  the console.

- Add timeries graph to 3di backend. See /3di/demo and watch the console.

- Refactored directory structure: blueprints now live in threedi_wms folder

- Added rasterinfo blueprint (app in Django lingo)

- Initial project structure created with nensskel 1.33.dev0.


