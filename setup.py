#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# module in "setup" mode. Stolen from `kplr`
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__CORONAGRAPH_SETUP__ = True
import coronagraph

long_description = \
    """Coronagraph noise model for directly imaging exoplanets."""

# Setup!
setup(name='coronagraph',
      version=coronagraph.__version__,
      description='Coronagraph noise model for directly imaging exoplanets.',
      long_description=long_description,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='http://github.com/jlustigy/coronagraph',
      author='Jacob Lustig-Yaeger',
      author_email='jlustigy@uw.edu',
      license='MIT',
      packages=['coronagraph'],
      install_requires=[
                        'numpy',
                        'scipy',
                        'matplotlib',
                        'numba',
                        'astropy'
                        ],
      dependency_links=[],
      scripts=[],
      include_package_data=True,
      zip_safe=False,
      data_files=["coronagraph/planets/ArcheanEarth_geo_albedo.txt",
                  "coronagraph/planets/EarlyMars_geo_albedo.txt",
                  "coronagraph/planets/EarlyVenus_geo_albedo.txt",
                  "coronagraph/planets/earth_avg_hitran2012_300_100000cm.trnst",
                  "coronagraph/planets/Hazy_ArcheanEarth_geo_albedo.txt",
                  "coronagraph/planets/Jupiter_geo_albedo.txt",
                  "coronagraph/planets/Mars_geo_albedo.txt",
                  "coronagraph/planets/Neptune_geo_albedo.txt",
                  "coronagraph/planets/Saturn_geo_albedo.txt",
                  "coronagraph/planets/Uranus_geo_albedo.txt",
                  "coronagraph/planets/Venus_geo_albedo.txt",
                  "coronagraph/planets/earth_quadrature_radiance_refl.dat"
                  ]
      )
