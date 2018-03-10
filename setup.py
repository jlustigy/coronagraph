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
                        'numba'],
      dependency_links=[],
      scripts=[],
      include_package_data=True,
      zip_safe=False
      )
