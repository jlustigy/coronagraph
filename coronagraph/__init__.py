#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Version number
__version__ = "0.0.2"

# Was coronagraph imported from setup.py?
try:
    __CORONAGRAPH_SETUP__
except NameError:
    __CORONAGRAPH_SETUP__ = False

if not __CORONAGRAPH_SETUP__:
    # This is a regular coronagraph run
    from .teleplanstar import *
    from . import observe
    from .observe import *
    from .degrade_spec import *
    from . import filters
    from .convolve_spec import *
    from .count_rates import *
    from .count_rates_wrapper import *
    from . import noise_routines
