#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
transit_demo.py |github|
------------------------

A demo for modeling transmission spectra with a blank mask coronagraph (i.e.
the coronagraph is not blocking the star's light).

  .. role:: raw-html(raw)
     :format: html

  .. |github| replace:: :raw-html:`<a href = "https://github.com/jlustigy/coronagraph/blob/master/scripts/transit_demo.py"><i class="fa fa-github" aria-hidden="true"></i></a>`

'''
from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)


# Import some standard python packages
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# The location to *this* file
RELPATH = os.path.dirname(__file__)

# Import coronagraph model
import coronagraph as cg
from coronagraph import plot_setup
plot_setup.setup()

def earth_analog_transits(d = 10., ntran = 10, nout = 2):
    '''
    Simulate the transmission spectrum of Earth transiting a Sun-like star that
    is `d` parsecs away.

    Parameters
    ----------
    d : float
        Distance to system [pc]
    ntran : int
        Number of transits
    nout : int
        Number of out-of-transit transit durations to observe

      .. plot::
        :align: center

        from scripts import transit_demo
        from coronagraph import plot_setup
        plot_setup.setup()
        transit_demo.earth_analog_transits()
    '''

    # Read-in high-res Earth model data
    lam, tdepth, fstar = cg.get_earth_spectrum()

    # Instantiate transit noise model
    tn = cg.TransitNoise(tdur = 8.0 * 60 * 60,
                         d = d,
                         r = 1.0,
                         Rp = 1.0,
                         Rs = 1.0,
                         ntran = ntran,
                         Tput = 0.5,
                         Res = 70,
                         nout = nout)

    # Calculate count rates
    tn.run_count_rates(lam, tdepth, fstar)

    # Plot the spectrum
    fig, ax = tn.plot_spectrum()
    plt.show()

    # Plot the SNR
    fig, ax = tn.plot_SNRn()
    plt.show()

    # Plot the number of transits to given SNR
    fig, ax = tn.plot_ntran_to_wantsnr()
    plt.show()

    # This is the SNR we want on the max difference in planet radius
    wantvsnr = 3
    # Calculate the SNR we want for the transit depths to get the right
    #   SNR on the radius difference
    wantsnr = wantvsnr * np.mean(tn.RpRs2) / (np.max(tn.RpRs2) - np.min(tn.RpRs2))
    tn.recalc_wantsnr(wantsnr = wantsnr)

    # Plot the number of transits to new SNR
    fig, ax = tn.plot_ntran_to_wantsnr()
    plt.show()

    # Plot the time to new SNR
    fig, ax = tn.plot_time_to_wantsnr()
    plt.show()

    # Plot the count rates
    fig, ax = tn.plot_count_rates()
    plt.show()

if __name__ == '__main__':

    earth_analog_transits()
