Changelog of threedi-wms
===================================================


0.8 (2014-02-17)
----------------

- Added option maxpoints to gettimeseries: throw away points till you have the max number of points you have :-)

- Add find link for netCDF4 1.0.4 to buildout.cfg.


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


