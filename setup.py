from setuptools import setup

version = '0.2.dev0'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'celery',
    'Flask',
    'gislib',
    'gunicorn',
    'matplotlib',
    'netCDF4',
    'Pillow',
    'requests',
    'scipy',
    'setuptools',
    'SQLAlchemy',
    'Shapely',
    ],

tests_require = [
    'nose',
    'coverage',
    ]

setup(name='threedi-wms',
      version=version,
      description="Flexible extensible wms server originally developed for 3di.",
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[],
      keywords=[],
      author='Arjan Verkerk',
      author_email='arjan.verkerk@nelen-schuurmans.nl',
      url='',
      license='GPL',
      packages=['threedi_wms'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={'test': tests_require},
      entry_points={
          'console_scripts': [
              'flask=server.app:run',
              'sandbox=threedi_wms.sandbox:main',
          ]},
      )
