#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Set the default plot style borrowing from Dan Foreman-Mackey's `celerite`.
"""

from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

from matplotlib import rcParams

def setup():
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["font.size"] = 20
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
