#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
luvoir_demo.py |github|
-----------------------

A demo simulation of Earth at 10 pc using a LUVOIR-like telescope setup.

  .. plot::
    :align: center

    from scripts import luvoir_demo
    luvoir_demo._test()

  .. role:: raw-html(raw)
     :format: html

  .. |github| replace:: :raw-html:`<a href = "https://github.com/jlustigy/coronagraph/blob/master/scripts/luvoir_demo.py"><i class="fa fa-github" aria-hidden="true"></i></a>`

'''
from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Import some standard python packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
import sys, os

# The location to *this* file
RELPATH = os.path.dirname(__file__)

# Import coronagraph model
import coronagraph as cg
from coronagraph import plot_setup

def _test():
    '''

    '''
    run()

def run():
    '''

    '''

    ################################
    # PARAMETERS
    ################################

    # Integration time (hours)
    Dt = 20.0

    # Planet params
    alpha = 90.     # phase angle at quadrature
    Phi   = cg.teleplanstar.lambertPhaseFunction(alpha)      # phase function at quadrature (already included in SMART run)
    Rp    = 1.0     # Earth radii
    r     = 1.0     # semi-major axis (AU)

    # Stellar params
    Teff  = 5780.   # Sun-like Teff (K)
    Rs    = 1.      # star radius in solar radii

    # Planetary system params
    d    = 10.     # distance to system (pc)
    Nez  = 1.      # number of exo-zodis

    # Plot params
    plot = True
    ref_lam = 0.55
    title = ""
    ylim =  [-0.1, 0.3]
    xlim =  None
    tag = ""

    # Save params
    savefile = False
    saveplot = False


    ################################
    # READ-IN DATA
    ################################

    # Read-in spectrum file
    fn = os.path.join(RELPATH, '../coronagraph/planets/earth_quadrature_radiance_refl.dat')
    model = np.loadtxt(fn, skiprows=8)
    lamhr = model[:,0]
    radhr = model[:,1]
    solhr = model[:,2]

    # Calculate hi-resolution reflectivity
    Ahr   = np.pi*(np.pi*radhr/solhr)

    ################################
    # RUN CORONAGRAPH MODEL
    ################################

    # Run coronagraph with default LUVOIR telescope (aka no keyword arguments)
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez,\
                       lammax=1.6)

    # Calculate background photon count rates
    cb = (cz + cez + csp + cD + cR + cth)

    # Convert hours to seconds
    Dts = Dt * 3600.

    # Calculate signal-to-noise assuming background subtraction (the "2")
    SNR  = cp*Dts/np.sqrt((cp + 2*cb)*Dts)

    # Calculate 1-sigma errors
    sig= Cratio/SNR

    # Add gaussian noise to flux ratio
    spec = Cratio + np.random.randn(len(Cratio))*sig

    ################################
    # PLOTTING
    ################################

    if plot:

        plot_setup.setup()

        # Create figure
        fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])

        # Set string for plot text
        if Dt > 2.0:
            timestr = "{:.0f}".format(Dt)+' hours'
        else:
            timestr = "{:.0f}".format(Dt*60)+' mins'
        plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'+\
        '\n Integration time = '+timestr

        # If a reference wavelength is specified then return the SNR at that wl
        # corresponding to the integration time given
        if ref_lam:
            ireflam = (np.abs(lam - ref_lam)).argmin()
            ref_SNR = SNR[ireflam]
            plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
                ' at '+"{:.2f}".format(lam[ireflam])+r' $\mu$m'

        # Draw plot
        ax.plot(lam, Cratio*1e9, lw=2.0, color="purple", alpha=0.7, ls="steps-mid")
        ax.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

        # Set labels
        ax.set_ylabel(r"F$_p$/F$_s$ ($\times 10^9$)", fontsize = 25)
        ax.set_xlabel("Wavelength [$\mu$m]", fontsize = 25)
        ax.set_title(title)
        ax.text(0.99, 0.99, plot_text,\
             verticalalignment='top', horizontalalignment='right',\
             transform=ax.transAxes,\
             color='black', fontsize=20)

        # Adjust x,y limits
        if ylim is not None: ax.set_ylim(ylim)
        if xlim is not None: ax.set_xlim(xlim)

        # Save plot if requested
        if saveplot:
            plot_tag = "luvoir_demo_"+title+tag+".png"
            fig.savefig(plot_tag)
            print('Saved: ' + plot_tag)
        else:
            plt.show()

    ################################
    # SAVING
    ################################

    # Save Synthetic data file (wavelength, albedo, error) if requested
    if savefile:
        data_tag = 'luvoir_demo_'+tag+'.txt'
        y_sav = np.array([lam,spec,sig])
        np.savetxt(data_tag, y_sav.T)
        print('Saved: ' + data_tag)

if __name__ == '__main__':

    run()
